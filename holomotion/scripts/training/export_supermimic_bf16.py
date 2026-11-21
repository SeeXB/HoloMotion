"""Export true BF16 CUDA MLPs and compare saved physical input histories."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from holomotion.scripts.training.export_supermimic import export as export_fp32
from holomotion.src.algo.supermimic import networks
from holomotion.src.deployment.bf16_policy import NativeBF16Policy
from holomotion.src.env.reference_tracking.mdp import unflatten_state
from holomotion.src.env.reference_tracking.task import load_task
from holomotion.src.utils.artifacts import atomic_json, fingerprint


def export(training, output):
    import onnx
    from onnx import TensorProto, helper

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(
            "Native BF16 export requires a BF16-capable CUDA GPU"
        )
    networks.AMP_DTYPE = None
    checkpoint = training / "checkpoint_final.pt"
    # Reuse the original export's input, archived-source and reconstruction
    # verification. The new artifact starts with all deployment gates unset.
    config = export_fp32(checkpoint, output)
    fp32_path = output / "policy_fp32_reference.onnx"
    (output / "policy.onnx").rename(fp32_path)
    bundle = torch.load(checkpoint, map_location="cpu", weights_only=False)
    policy = networks.Policy(
        config["obs_dim"],
        config["goal_dim"],
        config["history"],
        num_j=len(config["joint_names"]),
    ).eval()
    policy.load_state_dict(bundle["pi"], strict=True)
    policy = NativeBF16Policy(policy).cuda().eval()
    with np.load(output / "reference.npz", allow_pickle=False) as data:
        goal = torch.from_numpy(data["goals"][:1].copy()).cuda()
    history = torch.zeros(
        1, config["history"], config["obs_dim"], device="cuda"
    )
    path = output / "policy.onnx"
    with torch.no_grad():
        torch.onnx.export(
            policy,
            (history, goal),
            str(path),
            input_names=["obs_hist", "goal"],
            output_names=["action"],
            opset_version=17,
            dynamo=False,
        )
    graph = onnx.load(path)
    # This CUDA runtime implements BF16 GEMM, but not the newer BF16 Elu
    # schema. Preserve each activation's BF16 boundary with FP32 opmath.
    nodes = []
    for node in graph.graph.node:
        if node.op_type != "Elu":
            nodes.append(node)
            continue
        incoming, outgoing = node.input[0], node.output[0]
        nodes.append(
            helper.make_node(
                "Cast",
                [incoming],
                [incoming + "_elu_fp32"],
                to=TensorProto.FLOAT,
            )
        )
        node.input[0] = incoming + "_elu_fp32"
        node.output[0] = outgoing + "_elu_fp32"
        nodes.extend(
            [
                node,
                helper.make_node(
                    "Cast",
                    [outgoing + "_elu_fp32"],
                    [outgoing],
                    to=TensorProto.BFLOAT16,
                ),
            ]
        )
    del graph.graph.node[:]
    graph.graph.node.extend(nodes)
    onnx.checker.check_model(graph, full_check=True)
    onnx.save(graph, path)
    config.update(
        inference_precision="bf16_gemm_fp32_elu_bf16_boundary",
        execution_provider="CUDAExecutionProvider",
        onnx_graph_optimization="disabled",
        onnx_opset=17,
        onnx_max_abs_error=None,
        parity_samples=0,
        parity_reference="Saved physical histories, native CUDA BF16 policy",
        numerical_precision_passed=False,
        exported_simulation_passed=False,
        bf16_export_sources={
            str(p): fingerprint(p)
            for p in (
                Path(__file__),
                Path(__file__).resolve().parents[2]
                / "src/deployment/bf16_policy.py",
            )
        },
    )
    config["files"].update(
        {
            "policy.onnx": fingerprint(path),
            fp32_path.name: fingerprint(fp32_path),
        }
    )
    atomic_json(output / "manifest.json", config)
    return config


def validate(training, output):
    import onnxruntime as ort

    from holomotion.src.deployment.supermimic_r1 import OfflineActor

    torch.set_num_threads(8)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    bundle = torch.load(
        training / "checkpoint_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    args = bundle["specification"]["arguments"]
    actor = OfflineActor(output)
    _, motion, mdp = load_task(
        robot_urdf=Path(args["robot_urdf"]),
        manifest=Path(args["manifest"]),
        output=output / "parity_kinematics",
        device="cpu",
        profile=args["profile"],
    )
    policy = (
        networks.Policy(
            mdp.obs_dim,
            actor.config["goal_dim"],
            args["hist"],
            num_j=mdp.num_j,
        )
        .cuda()
        .eval()
    )
    policy.load_state_dict(bundle["pi"], strict=True)
    networks.AMP_DTYPE = torch.bfloat16
    trace = torch.load(
        training / "evaluation/rollout_final.pt",
        map_location="cpu",
        weights_only=False,
    )
    n, count = mdp.num_j, len(trace["action"])
    start = motion.clip_start
    initial = mdp.obs(
        motion.ref_state(start),
        torch.zeros(1, n),
        torch.zeros(1, n + 4),
        start,
    )
    observations = torch.cat(
        [
            initial,
            mdp.obs(
                unflatten_state(trace["next_state"][:-1]),
                trace["action"][:-1],
                torch.zeros(count - 1, n + 4),
                trace["reference_frame"][:-1],
            ),
        ]
    )
    frames = torch.cat([start, trace["reference_frame"][:-1]])
    goals = motion.goal(frames, args["goal_frames"])
    np.testing.assert_allclose(
        goals.numpy(), actor.goals[frames], atol=1e-5, rtol=1e-5
    )
    history = initial[:, None].repeat(1, args["hist"], 1)
    native, exported, recorded_errors = [], [], []
    with torch.no_grad():
        for i in range(count):
            if i:
                history = torch.cat(
                    [history[:, 1:], observations[i : i + 1, None]], 1
                )
            # Feed exactly the deployment bundle's stored FP32 goals to both.
            goal = actor.goals[frames[i] : frames[i] + 1]
            expected = (
                policy(history.cuda(), torch.from_numpy(goal).cuda())
                .cpu()
                .numpy()
            )
            actual = actor.session.run(
                None, dict(obs_hist=history.numpy(), goal=goal)
            )[0]
            native.append(expected[0])
            exported.append(actual[0])
            recorded_errors.append(
                float(abs(expected[0] - trace["action"][i].numpy()).max())
            )
    native, exported = np.asarray(native), np.asarray(exported)
    errors = abs(native - exported).max(axis=-1)
    samples_path = output / "bf16_precision_samples.npz"
    np.savez_compressed(
        samples_path,
        reference_frames=frames.numpy(),
        native_actions=native,
        exported_actions=exported,
    )
    report = dict(
        samples=count,
        max_abs_action_error=float(errors.max()),
        exact_frames=int((errors == 0).sum()),
        frames_above_2e4=int((errors > 2e-4).sum()),
        worst_reference_frame=int(frames[errors.argmax()]),
        native_vs_recorded_max_abs_error=max(recorded_errors),
        tolerance=2e-4,
        passed=bool(errors.max() <= 2e-4),
        execution_providers=actor.session.get_providers(),
        provider_options=actor.session.get_provider_options(),
        onnxruntime_version=ort.__version__,
        torch_version=torch.__version__,
        gpu=torch.cuda.get_device_name(),
        checkpoint_sha256=actor.config["source_checkpoint_sha256"],
        artifact_sha256=actor.config["files"]["policy.onnx"],
        per_frame_max_abs_errors=errors.tolist(),
    )
    atomic_json(output / "bf16_precision_report.json", report)
    config = actor.config
    config.update(
        onnx_max_abs_error=report["max_abs_action_error"],
        parity_samples=count,
        numerical_precision_passed=report["passed"],
    )
    config["files"][samples_path.name] = fingerprint(samples_path)
    atomic_json(output / "manifest.json", config)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--export-only", action="store_true")
    mode.add_argument("--validate-existing", action="store_true")
    args = parser.parse_args()
    training, output = args.training.resolve(), args.output.resolve()
    if not args.validate_existing:
        export(training, output)
    if not args.export_only:
        result = validate(training, output)
        print(
            json.dumps(
                {
                    k: v
                    for k, v in result.items()
                    if k != "per_frame_max_abs_errors"
                },
                indent=2,
            )
        )

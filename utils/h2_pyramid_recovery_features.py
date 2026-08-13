"""Source- and truth-free feature assembly for H2 component recovery V2.

The public builders consume only already-sampled, event-aligned model features,
event locations, complete M20 components, and the Stage1 retained-core mask.
They deliberately have no source identity, path, fold, label, or target input.

One output row represents one complete-component/time-bin slice.  Its first 86
values are node means of model-derived event features and its final 10 values
are the relative geometry/trajectory features frozen in the V2 protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch


DECODER_CHANNELS = 16
TEMPORAL_SCALES = (16, 32, 64, 160)
SCALE_CHANNELS = 16
CENTRE_CHANNELS = 3
TEMPORAL_BIN_SIZE = 50
MAX_COMPONENTS_PER_BATCH = 64
MAX_TEMPORAL_NODES = 160

EVENT_FEATURE_DIM = (
    DECODER_CHANNELS
    + len(TEMPORAL_SCALES) * SCALE_CHANNELS
    + 3
    + CENTRE_CHANNELS
)
RELATIVE_FEATURE_DIM = 10
NODE_FEATURE_DIM = EVENT_FEATURE_DIM + RELATIVE_FEATURE_DIM

DECODER_FEATURE_SLICE = slice(0, 16)
SCALE_CONTEXT_FEATURE_SLICE = slice(16, 80)
LOGIT_FEATURE_SLICE = slice(80, 83)
CENTRE_FEATURE_SLICE = slice(83, 86)
RELATIVE_FEATURE_SLICE = slice(86, 96)

EVENT_FEATURE_NAMES = (
    tuple(f"M20_decoder0_mean_{channel:02d}" for channel in range(16))
    + tuple(
        f"scale_{scale}_encoded_context_mean_{channel:02d}"
        for scale in TEMPORAL_SCALES
        for channel in range(16)
    )
    + (
        "M20_logit_mean",
        "Stage1_logit_mean",
        "Stage1_minus_M20_logit_mean",
        "centre_negative_mean",
        "centre_positive_mean",
        "centre_activity_mean",
    )
)

RELATIVE_FEATURE_NAMES = (
    "log1p_node_event_count",
    "node_event_fraction_within_component",
    "relative_time_from_component_midpoint_normalized_by_component_span",
    "relative_centroid_x",
    "relative_centroid_y",
    "relative_bbox_width",
    "relative_bbox_height",
    "previous_node_centroid_delta_x",
    "previous_node_centroid_delta_y",
    "Stage1_retained_core_fraction",
)

NODE_FEATURE_NAMES = EVENT_FEATURE_NAMES + RELATIVE_FEATURE_NAMES

if len(EVENT_FEATURE_NAMES) != EVENT_FEATURE_DIM:
    raise RuntimeError("event feature schema does not contain exactly 86 values")
if len(NODE_FEATURE_NAMES) != NODE_FEATURE_DIM:
    raise RuntimeError("node feature schema does not contain exactly 96 values")


def _numpy_cpu(values, description):
    if isinstance(values, torch.Tensor):
        if values.device.type != "cpu":
            raise ValueError(f"{description} must remain on CPU")
        tensor = values.detach()
        if tensor.dtype == torch.bfloat16:
            tensor = tensor.to(dtype=torch.float32)
        return tensor.numpy()
    return np.asarray(values)


def _finite_float(values, description):
    try:
        result = np.asarray(_numpy_cpu(values, description), dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be numeric") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{description} must be finite")
    return result


def _event_matrix(values, event_count, width, description):
    result = _finite_float(values, description)
    if result.shape != (event_count, width):
        raise ValueError(f"{description} must have shape [{event_count},{width}]")
    return result


def _event_vector(values, event_count, description):
    result = _finite_float(values, description)
    if result.shape == (event_count, 1):
        result = result[:, 0]
    if result.shape != (event_count,):
        raise ValueError(f"{description} must have shape [{event_count}]")
    return result


def assemble_event_feature_rows(
    decoder_rows,
    scale_context_rows,
    base_logits,
    stage1_logits,
    centre_rows,
):
    """Assemble the 86 event-derived values in their frozen protocol order.

    ``scale_context_rows`` is ordered ``[event, scale, channel]`` with scales
    ``(16, 32, 64, 160)``.  The signed logit delta is Stage1 minus M20/base.
    """

    decoder = _finite_float(decoder_rows, "decoder_rows")
    if decoder.ndim != 2 or decoder.shape[1] != DECODER_CHANNELS:
        raise ValueError("decoder_rows must have shape [E,16]")
    event_count = int(decoder.shape[0])
    scale_context = _finite_float(scale_context_rows, "scale_context_rows")
    expected_scale_shape = (event_count, len(TEMPORAL_SCALES), SCALE_CHANNELS)
    if scale_context.shape != expected_scale_shape:
        raise ValueError("scale_context_rows must have shape [E,4,16]")
    base = _event_vector(base_logits, event_count, "base_logits")
    stage1 = _event_vector(stage1_logits, event_count, "stage1_logits")
    centre = _event_matrix(
        centre_rows, event_count, CENTRE_CHANNELS, "centre_rows"
    )
    result = np.concatenate(
        (
            decoder,
            scale_context.reshape(event_count, -1),
            base[:, None],
            stage1[:, None],
            (stage1 - base)[:, None],
            centre,
        ),
        axis=1,
    )
    if result.shape != (event_count, EVENT_FEATURE_DIM):
        raise RuntimeError("assembled event feature schema mismatch")
    return result.astype(np.float32, copy=False)


def _location_columns(locations, event_count):
    coordinates = _finite_float(locations, "locations")
    if coordinates.ndim != 2 or coordinates.shape[0] != event_count:
        raise ValueError("locations must align with event feature rows")
    if coordinates.shape[1] == 3:
        batch = np.zeros(event_count, dtype=np.int64)
        xy = coordinates[:, :2]
        times = coordinates[:, 2]
    elif coordinates.shape[1] >= 4:
        batch = coordinates[:, 0]
        xy = coordinates[:, 1:3]
        times = coordinates[:, 3]
    else:
        raise ValueError("locations must be [E,3] x/y/t or [E,4+] batch/x/y/t")
    integer_values = np.column_stack((batch, xy, times))
    if not np.equal(integer_values, np.rint(integer_values)).all():
        raise ValueError("location coordinates and timestamps must be integer-valued")
    return (
        np.rint(batch).astype(np.int64),
        np.rint(xy).astype(np.int64),
        np.rint(times).astype(np.int64),
    )


def _component_rows(components, event_count):
    if hasattr(components, "event_indices"):
        components = components.event_indices
    try:
        raw_components = tuple(components)
    except TypeError as error:
        raise TypeError("components must be a sequence of complete components") from error
    if len(raw_components) > MAX_COMPONENTS_PER_BATCH:
        raise ValueError("component count exceeds the frozen microbatch bound of 64")
    normalized = []
    assigned = np.zeros(event_count, dtype=np.bool_)
    for values in raw_components:
        raw = _numpy_cpu(values, "components")
        if raw.dtype.kind not in "iu" or raw.ndim != 1 or raw.size == 0:
            raise ValueError("every component must be a non-empty integer vector")
        rows = np.sort(raw.astype(np.int64, copy=False))
        if rows[0] < 0 or rows[-1] >= event_count:
            raise ValueError("a component refers outside the event rows")
        if np.unique(rows).size != rows.size:
            raise ValueError("a component contains duplicate event rows")
        if np.any(assigned[rows]):
            raise ValueError("complete components must be disjoint")
        assigned[rows] = True
        normalized.append(rows)
    return tuple(normalized)


def _retained_vector(values, event_count):
    retained = _finite_float(values, "stage1_retained")
    if retained.shape == (event_count, 1):
        retained = retained[:, 0]
    if retained.shape != (event_count,):
        raise ValueError("stage1_retained must have shape [E]")
    if not np.logical_or(retained == 0.0, retained == 1.0).all():
        raise ValueError("stage1_retained must contain only Boolean/zero-one values")
    return retained


def assemble_component_time_nodes(
    event_rows,
    locations,
    components,
    stage1_retained,
):
    """Return one deterministic ``[node,96]`` array per complete component.

    Component events are canonically sorted before reduction, and temporal
    nodes are ordered from early to late.  Geometry is expressed only in the
    component's own coordinate system.  Consequently, event order within a
    node and an integral 50-tick translation of all timestamps cannot change
    the resulting values.
    """

    features = _finite_float(event_rows, "event_rows")
    if features.ndim != 2 or features.shape[1] != EVENT_FEATURE_DIM:
        raise ValueError("event_rows must have shape [E,86]")
    event_count = int(features.shape[0])
    batches, xy, times = _location_columns(locations, event_count)
    retained = _retained_vector(stage1_retained, event_count)
    normalized_components = _component_rows(components, event_count)

    output = []
    for rows in normalized_components:
        if np.unique(batches[rows]).size != 1:
            raise ValueError("one complete component cannot cross batch values")
        event_bins = np.floor_divide(times[rows], TEMPORAL_BIN_SIZE)
        temporal_bins = np.unique(event_bins)
        if temporal_bins.size > MAX_TEMPORAL_NODES:
            raise ValueError("a component exceeds the frozen 160-node bound")

        component_xy = xy[rows].astype(np.float64, copy=False)
        component_centroid = component_xy.mean(axis=0, dtype=np.float64)
        component_minimum = component_xy.min(axis=0)
        component_maximum = component_xy.max(axis=0)
        component_extent = component_maximum - component_minimum + 1.0
        component_event_count = float(rows.size)
        first_bin = int(temporal_bins[0])
        last_bin = int(temporal_bins[-1])
        component_span = max(last_bin - first_bin, 1)
        component_midpoint = 0.5 * (first_bin + last_bin)

        node_rows = []
        previous_centroid = None
        for temporal_bin in temporal_bins:
            member_rows = rows[event_bins == temporal_bin]
            member_xy = xy[member_rows].astype(np.float64, copy=False)
            centroid = member_xy.mean(axis=0, dtype=np.float64)
            extent = member_xy.max(axis=0) - member_xy.min(axis=0) + 1.0
            if previous_centroid is None:
                previous_delta = np.zeros(2, dtype=np.float64)
            else:
                previous_delta = (centroid - previous_centroid) / component_extent
            geometry = np.asarray(
                (
                    np.log1p(float(member_rows.size)),
                    float(member_rows.size) / component_event_count,
                    (float(temporal_bin) - component_midpoint) / component_span,
                    (centroid[0] - component_centroid[0]) / component_extent[0],
                    (centroid[1] - component_centroid[1]) / component_extent[1],
                    extent[0] / component_extent[0],
                    extent[1] / component_extent[1],
                    previous_delta[0],
                    previous_delta[1],
                    retained[member_rows].mean(dtype=np.float64),
                ),
                dtype=np.float64,
            )
            event_mean = features[member_rows].mean(axis=0, dtype=np.float64)
            node_rows.append(np.concatenate((event_mean, geometry)))
            previous_centroid = centroid
        component_nodes = np.asarray(node_rows, dtype=np.float32)
        if component_nodes.shape != (temporal_bins.size, NODE_FEATURE_DIM):
            raise RuntimeError("component node feature schema mismatch")
        if not np.isfinite(component_nodes).all():
            raise RuntimeError("component node features must be finite")
        output.append(component_nodes)
    return tuple(output)


def build_component_node_features(
    decoder_rows,
    scale_context_rows,
    base_logits,
    stage1_logits,
    centre_rows,
    stage1_retained,
    locations,
    components,
):
    """Convenience wrapper for the complete source-/truth-free V2 assembly."""

    event_rows = assemble_event_feature_rows(
        decoder_rows,
        scale_context_rows,
        base_logits,
        stage1_logits,
        centre_rows,
    )
    return assemble_component_time_nodes(
        event_rows,
        locations,
        components,
        stage1_retained,
    )


def collate_component_nodes(component_nodes):
    """Zero-pad deterministic component sequences for the recovery head.

    Returns CPU ``float32`` node features and a CPU Boolean validity mask.
    Padding length is the longest sequence in this batch and padding bits are
    always exactly positive zero.
    """

    if not isinstance(component_nodes, Sequence):
        raise TypeError("component_nodes must be a sequence")
    values = tuple(component_nodes)
    if not values:
        raise ValueError("at least one component is required for collation")
    if len(values) > MAX_COMPONENTS_PER_BATCH:
        raise ValueError("component count exceeds the frozen microbatch bound of 64")
    normalized = []
    for sequence in values:
        array = _finite_float(sequence, "component_nodes")
        if array.ndim != 2 or array.shape[1] != NODE_FEATURE_DIM:
            raise ValueError("each component sequence must have shape [N,96]")
        if array.shape[0] < 1 or array.shape[0] > MAX_TEMPORAL_NODES:
            raise ValueError("component sequence length must be in [1,160]")
        normalized.append(array.astype(np.float32, copy=False))
    maximum_nodes = max(array.shape[0] for array in normalized)
    padded = np.zeros(
        (len(normalized), maximum_nodes, NODE_FEATURE_DIM), dtype=np.float32
    )
    mask = np.zeros((len(normalized), maximum_nodes), dtype=np.bool_)
    for row, array in enumerate(normalized):
        length = int(array.shape[0])
        padded[row, :length] = array
        mask[row, :length] = True
    return torch.from_numpy(padded), torch.from_numpy(mask)


__all__ = (
    "CENTRE_FEATURE_SLICE",
    "DECODER_FEATURE_SLICE",
    "EVENT_FEATURE_DIM",
    "EVENT_FEATURE_NAMES",
    "LOGIT_FEATURE_SLICE",
    "MAX_COMPONENTS_PER_BATCH",
    "MAX_TEMPORAL_NODES",
    "NODE_FEATURE_DIM",
    "NODE_FEATURE_NAMES",
    "RELATIVE_FEATURE_NAMES",
    "RELATIVE_FEATURE_SLICE",
    "SCALE_CONTEXT_FEATURE_SLICE",
    "TEMPORAL_BIN_SIZE",
    "assemble_component_time_nodes",
    "assemble_event_feature_rows",
    "build_component_node_features",
    "collate_component_nodes",
)

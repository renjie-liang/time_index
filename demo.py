
def compute_overlap(pred, gt):
    # check format
    assert isinstance(pred, list) and isinstance(gt, list)
    pred_is_list = isinstance(pred[0], list)
    gt_is_list = isinstance(gt[0], list)
    pred = pred if pred_is_list else [pred]
    gt = gt if gt_is_list else [gt]
    # compute overlap
    pred, gt = np.array(pred), np.array(gt)
    inter_left = np.maximum(pred[:, 0, None], gt[None, :, 0])
    inter_right = np.minimum(pred[:, 1, None], gt[None, :, 1])
    inter = np.maximum(0.0, inter_right - inter_left)
    union_left = np.minimum(pred[:, 0, None], gt[None, :, 0])
    union_right = np.maximum(pred[:, 1, None], gt[None, :, 1])
    union = np.maximum(1e-12, union_right - union_left)
    overlap = 1.0 * inter / union
    # reformat output
    overlap = overlap if gt_is_list else overlap[:, 0]
    overlap = overlap if pred_is_list else overlap[0]
    return overlap


# def time_to_index(st, num_units, duration):
#     start_time, end_time = st
#     s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
#     e_times = np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
#     candidates = np.stack([np.repeat(s_times[:, None], repeats=num_units, axis=1),
#                            np.repeat(e_times[None, :], repeats=num_units, axis=0)], axis=2).reshape((-1, 2))
#     overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(num_units, num_units)
#     start_index = np.argmax(overlaps) // num_units
#     end_index = np.argmax(overlaps) % num_units
#     return start_index, end_index


# def index_to_time(st, num_units, duration):
#     start_index, end_index = st
#     s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
#     e_times = np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
#     start_time = s_times[start_index]
#     end_time = e_times[end_index]
#     return start_time, end_time

# def time_to_index(t, duration, vlen):
#     if isinstance(t, list):
#         res = []
#         for i in t:
#             res.append(time_to_index(i, duration, vlen))
#         return res
#     else:
#         return round(t / duration * (vlen - 1))

def time_to_index(start_time, end_time, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) / float(num_units) * duration
    e_times = np.arange(1, num_units + 1).astype(np.float32) / float(num_units) * duration
    candidates = np.stack([np.repeat(s_times[:, None], repeats=num_units, axis=1),
                           np.repeat(e_times[None, :], repeats=num_units, axis=0)], axis=2).reshape((-1, 2))
    overlaps = compute_overlap(candidates.tolist(), [start_time, end_time]).reshape(num_units, num_units)
    start_index = np.argmax(overlaps) // num_units
    end_index = np.argmax(overlaps) % num_units
    return start_index, end_index, overlaps


def index_to_time(start_index, end_index, num_units, duration):
    s_times = np.arange(0, num_units).astype(np.float32) * duration / float(num_units)
    e_times = np.arange(1, num_units + 1).astype(np.float32) * duration / float(num_units)
    start_time = s_times[start_index]
    end_time = e_times[end_index]
    return start_time, end_time


# def index_to_time(t, duration, vlen):
#     if isinstance(t, list):
#         res = []
#         for i in t:
#             res.append(index_to_time(i, duration, vlen))
#         return res
#     else:
#         return round(t / (vlen-1) * duration, 2)

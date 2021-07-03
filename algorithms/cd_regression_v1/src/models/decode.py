import torch
import torch.nn as nn
from .utils import _gather_feat, _tranpose_and_gather_feat
from .utils import _nms, _topk, _topk_channel

def generic_decode(output, K=100, opt=None):
	if not ('hm' in output):
		return {}
	
	heat = output['hm']
	batch, cat, height, width = heat.size()

	heat = _nms(heat)
	scores, inds, clses, ys0, xs0 = _topk(heat, K=K)

	clses  = clses.view(batch, K)
	scores = scores.view(batch, K)
	bboxes = None
	cts = torch.cat([xs0.unsqueeze(2), ys0.unsqueeze(2)], dim=2)
	ret = {'scores': scores, 'clses': clses.float(), 
				 'xs': xs0, 'ys': ys0, 'cts': cts}
	if 'reg' in output:
		reg = output['reg']
		reg = _tranpose_and_gather_feat(reg, inds)
		reg = reg.view(batch, K, 2)
		xs = xs0.view(batch, K, 1) + reg[:, :, 0:1]
		ys = ys0.view(batch, K, 1) + reg[:, :, 1:2]
	else:
		xs = xs0.view(batch, K, 1) + 0.5
		ys = ys0.view(batch, K, 1) + 0.5

	if 'wh' in output:
		wh = output['wh']
		wh = _tranpose_and_gather_feat(wh, inds) # B x K x (F)
		# wh = wh.view(batch, K, -1)
		wh = wh.view(batch, K, 2)
		wh[wh < 0] = 0
		if wh.size(2) == 2 * cat: # cat spec
			wh = wh.view(batch, K, -1, 2)
			cats = clses.view(batch, K, 1, 1).expand(batch, K, 1, 2)
			wh = wh.gather(2, cats.long()).squeeze(2) # B x K x 2
		else:
			pass
		bboxes = torch.cat([xs - wh[..., 0:1] / 2, 
												ys - wh[..., 1:2] / 2,
												xs + wh[..., 0:1] / 2, 
												ys + wh[..., 1:2] / 2], dim=2)
		ret['bboxes'] = bboxes
		# print('ret bbox', ret['bboxes'])
 
	regression_heads = ['tracking', 'dep']

	for head in regression_heads:
		if head in output:
			ret[head] = _tranpose_and_gather_feat(
				output[head], inds).view(batch, K, -1)

	if 'pre_inds' in output and output['pre_inds'] is not None:
		pre_inds = output['pre_inds'] # B x pre_K
		pre_K = pre_inds.shape[1]
		pre_ys = (pre_inds / width).int().float()
		pre_xs = (pre_inds % width).int().float()

		ret['pre_cts'] = torch.cat(
			[pre_xs.unsqueeze(2), pre_ys.unsqueeze(2)], dim=2)
	
	return ret

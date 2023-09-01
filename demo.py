import os
import cv2
from SegTracker import SegTracker
from model_args import aot_args, sam_args, segtracker_args
from PIL import Image, ImageDraw, ImageFont
from aot_tracker import _palette
import numpy as np
import torch
import imageio
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import gc
import argparse
import sys
import math
import json
import datetime
import pathlib
import time
from datetime import timedelta


PNG_COMPRESSION_LEVEL = 9
AVERAGE_DISTANCE = 21
POINT_COORDINATES = tuple([634, 985])


def seg_acc_click(Seg_Tracker, prompt, origin_frame):
	# seg acc to click
	predicted_mask, masked_frame = Seg_Tracker.seg_acc_click(
		origin_frame=origin_frame,
		coords=np.array(prompt["points_coord"]),
		modes=np.array(prompt["points_mode"]),
		multimask=prompt["multimask"],
	)
	Seg_Tracker = SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask)
	return predicted_mask #masked_frame


def SegTracker_add_first_frame(Seg_Tracker, origin_frame, predicted_mask):
	with torch.cuda.amp.autocast():
		# Reset the first frame's mask
		# frame_idx = 0
		Seg_Tracker.restart_tracker()
		Seg_Tracker.add_reference(origin_frame, predicted_mask) #frame_idx
		Seg_Tracker.first_frame_mask = predicted_mask
	return Seg_Tracker


def get_datetime(filename, videofile_type):
	try:
		if videofile_type=='facade_static':
			datetime_str = os.path.splitext(filename)[0].split('--')[0].split('_')[-1]
			return datetime.datetime.strptime(datetime_str, '%Y%m%d-%H%M%S')
		elif videofile_type=='facade_mobile':
			datetime_str = os.path.splitext(filename)[0].split('_')[1].lstrip('0')
			return datetime.datetime.strptime(datetime_str, '%Y%m%d%H%M%S')
		else:
			raise NotImplementedError
		return datetime_str
	except Exception as error:
		print("Невозможно определить время и дату видеоряда:", error)


def select_videos_between(path_to_videos, start_datetime, end_datetime, videofile_type='facade_static'):
	paths = sorted(pathlib.Path(path_to_videos).iterdir(),
		key=lambda x: get_datetime(x, videofile_type))
	in_between_dates = []
	start_date_dt = datetime.datetime.strptime(start_datetime, '%Y-%m-%dT%H:%M:%S')
	end_date_dt = datetime.datetime.strptime(end_datetime, '%Y-%m-%dT%H:%M:%S')
	for d in paths:
		ddt = get_datetime(d, videofile_type)
		if ddt >= start_date_dt and ddt <= end_date_dt:
			in_between_dates.append(d)
	return in_between_dates


def save_prediction(pred_mask,output_dir,file_name):
	save_mask = Image.fromarray(pred_mask.astype(np.uint8))
	save_mask = save_mask.convert(mode='P')
	save_mask.putpalette(_palette)
	save_mask.save(os.path.join(output_dir,file_name))


def colorize_mask(pred_mask):
	save_mask = Image.fromarray(pred_mask.astype(np.uint8))
	save_mask = save_mask.convert(mode='P')
	save_mask.putpalette(_palette)
	save_mask = save_mask.convert(mode='RGB')
	return np.array(save_mask)


def draw_mask(img, mask, alpha=0.5, id_countour=False):
	img_mask = np.zeros_like(img)
	img_mask = img
	if id_countour:
		# very slow ~ 1s per image
		obj_ids = np.unique(mask)
		obj_ids = obj_ids[obj_ids!=0]

		for id in obj_ids:
			# Overlay color on  binary mask
			if id <= 255:
				color = _palette[id*3:id*3+3]
			else:
				color = [0,0,0]
			foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
			binary_mask = (mask == id)

			# Compose image
			img_mask[binary_mask] = foreground[binary_mask]

			countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
			img_mask[countours, :] = 0
	else:
		binary_mask = (mask != 0)
		countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
		foreground = img * (1 - alpha) + colorize_mask(mask) * alpha
		img_mask[binary_mask] = foreground[binary_mask]
		img_mask[countours,:] = 0
	return img_mask.astype(img.dtype)


def draw_circle(img, center, radius, alpha=0.5):
	img_mask = np.zeros_like(img)
	img_mask = img

	x, y = np.indices((img.shape[0], img.shape[1]))
	mask = (np.hypot(center[0] - x, center[1] - y) - radius < 0.5).astype(int)

	binary_mask = (mask != 0)
	countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
	color = _palette[6:9]
	foreground = img * (1 - alpha) + np.ones_like(img) * alpha * np.array(color)
	img_mask[binary_mask] = foreground[binary_mask]
	img_mask[countours, :] = 0
	return img_mask.astype(img.dtype)


def draw_text(img: np.array, pos, text: str):
	im = Image.fromarray(img)
	font = ImageFont.load_default()
	draw = ImageDraw.Draw(im)
	draw.text(pos, text, fill='rgb(0, 0, 0)', font=font)
    return np.asarray(im)


def process_video(video_path, timestamp):
	torch.cuda.empty_cache()
	gc.collect()
	
	frame_idx = 0
	segtracker = SegTracker(segtracker_args, sam_args, aot_args)
	segtracker.restart_tracker()

	cap = cv2.VideoCapture(video_path)
	fps = cap.get(cv2.CAP_PROP_FPS)

	total_amount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	total_duration = total_amount / fps

	if frequency >= total_duration or frequency == 0:
		frame_poses = [0, total_amount - 1]
	elif frequency == -1:
		step = 1
	else:
		step = math.floor(total_amount * frequency / total_duration)
	frame_poses = [i for i in range(0, total_amount, step)]
	frame_poses = sorted(frame_poses)

	start_mask = []
	summ_start = 0

	counter = 0
	placed = False
	distance = 0
	distances = []

	with torch.cuda.amp.autocast():
		for i in frame_poses:
			cap.set(cv2.CAP_PROP_POS_FRAMES, i)
			ret, frame = cap.read()
			if not ret:
				break
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
			if frame_idx == 0:
				if args.frame is True:
					start_frame = cv2.imread(args.prompt + '/frame.png', cv2.IMREAD_UNCHANGED)
					pred_mask = seg_acc_click(segtracker, click_prompt, start_frame)
					pred_mask = segtracker.track(frame, update_memory=True)
				else:
					pred_mask = seg_acc_click(segtracker, click_prompt, frame)
				start_mask = pred_mask
				summ_start = start_mask.sum()
				prev_mask = pred_mask
				summ_prev = prev_mask.sum()
			else:
				pred_mask = segtracker.track(frame, update_memory=True)

			torch.cuda.empty_cache()
			gc.collect()

			if args.show_masks is True:
				save_prediction(pred_mask, args.output + '/masks', str(i) + '.png')

				if placed is False and pred_mask[POINT_COORDINATES] == 1:
					placed = True
					counter += 1
					distance = 1
				elif placed is True and pred_mask[POINT_COORDINATES] == 1:
					distance += 1
				elif placed is True and pred_mask[POINT_COORDINATES] == 0 and distance > 0.7 * AVERAGE_DISTANCE:
					placed = False
					distances.append({distance: str(round(i / fps))})
					distance = 0
				elif distance >= AVERAGE_DISTANCE:
					placed = False
					distances.append({distance: str(round(i / fps)) + 'F'})
					distance = 0
				else:
					placed = False

				mask_frame = draw_mask(frame, pred_mask)

				if placed is True:
					mask_frame = draw_circle(frame, [POINT_COORDINATES], 50)

				mask_frame = cv2.flip(mask_frame, 0)
				cv2.putText(image=mask_frame,
							text='Количество мешков: ' + str(counter),
							org=(332, 120),
							fontFace=cv2.FONT_HERSHEY_COMPLEX,
							fontScale=10,
							color=(0, 255, 0),
							thickness=2,
							lineType=cv2.LINE_AA,
							bottomLeftOrigin=True)
				mask_frame = cv2.flip(mask_frame, 0)
				
				# mask_frame = draw_text(mask_frame, (332, 120), 'Количество мешков: ' + str(counter))

				mask_frame = cv2.cvtColor(mask_frame, cv2.COLOR_RGB2BGR)
				cv2.imwrite(args.output + '/over/' + str(i) + '.png',
							mask_frame,
							[int(cv2.IMWRITE_PNG_COMPRESSION), PNG_COMPRESSION_LEVEL])

				print("processed frame {}".format(frame_idx), end='\r')
				frame_idx += 1
				continue

			summ_mask = pred_mask.sum()
			diff = pred_mask - prev_mask
			diff[np.where(diff != 1)] = 0
			summ_diff = diff.sum()
			percentage = '+' if summ_mask >= summ_prev else '-'
			if summ_prev != 0:
				percentage += str(round(100 * summ_diff / summ_prev))
			else:
				percentage += '100'
			diff_frame = draw_mask(frame, diff)
			diff_frame = cv2.cvtColor(diff_frame, cv2.COLOR_RGB2BGR)
			
			frame_timestapm = timestamp + timedelta(seconds=round(i / fps))
			frame_timestapm = frame_timestapm.strftime("%Y-%m-%dT%H:%M:%S")
			cv2.imwrite(args.output + '/' + str(frame_timestapm) + '_' + percentage + '.png',
						diff_frame,
						[int(cv2.IMWRITE_PNG_COMPRESSION), PNG_COMPRESSION_LEVEL])

			if frame_idx != len(frame_poses) - 2:
				prev_mask = pred_mask
				summ_prev = summ_mask
			else:
				prev_mask = start_mask
				summ_prev = summ_start

			print("processed frame {}, obj_num {}".format(frame_idx,segtracker.get_obj_num()),end='\r')
			frame_idx += 1
		cap.release()
		print('\nfinished')

		print('\nDistances:', distances)

	# manually release memory (after cuda out of memory)
	del segtracker
	torch.cuda.empty_cache()
	gc.collect()


parser = argparse.ArgumentParser(prog='Tracking',
								 description='Распознает изменения заданных объектов на видеоряде.')

parser.add_argument('input', help='Путь к папке с входными видео файлами.', type=str)
parser.add_argument('output', help='Путь к папке для выходных файлов.', type=str)
parser.add_argument('-v', '--frequency', help='Частота дискретизации - шаг в секундах, с которым обрабатываются кадры.\
					Например, для обработки одного кадра каждые 10 минут видео, задайте параметр равным 600.', type=float)
parser.add_argument('prompt', help='Путь к папке с файлами запроса, определяющими отслеживаемый объект.\
					Запрос должен быть в файле prompt.json.', type=str)
parser.add_argument('-f', '--frame', action='store_true', help='Параметр, указывающий на наличие отдельного кадра для входящего запроса в указанной директории.\
					Имя файла должно быть frame.png.')
parser.add_argument('-st', '--start_time', help="Дата и время начинала обработки,\
					в формате 'YYYY-MM-DDTHH:MM:SS', например '2023-07-21T01:05:21', что значит 1 час, 5 минут и 21 секунду для видео снятого 2023-07-21", type=str, default=None)
parser.add_argument('-et', '--end_time', help="Дата и время конца обработки,\
					в формате 'YYYY-MM-DDTHH:MM:SS', например '2023-07-21T01:05:21', что значит 1 час, 5 минут и 21 секунду для видео снятого 2023-07-21", type=str, default=None)
parser.add_argument('-vt', '--video_type', help="Тип видео. Один из:\
					facade_static - видео со статичной камеры на фасад,\
					facade_mobile - видео с мобильного регистратора в зоне фасада", type=str, default="facade_static=")
parser.add_argument('-sm', '--show_masks', help="Сохранять распознанные маски, а не только разницу между кадрами.", action='store_true')

if __name__ == '__main__':
	args = parser.parse_args()

	if args.frequency is not None:
		frequency = args.frequency
	else:
		frequency = 0

	if not os.path.exists(os.path.dirname(args.prompt)):
		raise Exception('Wrong prompt path.')

	if args.frame is True:
		custom_frame = args.prompt + '/frame.png'

	if not os.path.exists(os.path.dirname(args.input)):
		raise Exception('Wrong input path.')

	if not os.path.isdir(args.output):
		os.mkdir(args.output)

	if args.show_masks is True and not os.path.isdir(args.output + '/masks'):
		os.mkdir(args.output + '/masks')

	if args.show_masks is True  and not os.path.isdir(args.output + '/over'):
		os.mkdir(args.output + '/over')

	with open(args.prompt + '/prompt.json') as file:
		click_prompt = json.load(file)

	if args.start_time is None or args.end_time is None or args.video_type is None:
		raise Exception('Wrong datetime settings.')

	sam_gap = segtracker_args['sam_gap']

	videolist = select_videos_between(args.input, args.start_time, args.end_time, args.video_type)
	videolist_len = len(videolist)

	if videolist:
		for path in videolist:
			process_video(video_path=str(path), timestamp=get_datetime(path, args.video_type))
			videolist_len -= 1
			print(f"Осталось видео: {videolist_len}")
	else:
		print("В заданный интервал не попадает ни одно видео.")


	# save 'done' file marking the end of the process
	f = open(args.output + "/done", "w")
	f.close()


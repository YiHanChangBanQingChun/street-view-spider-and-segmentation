
import requests
import json
from PIL import Image
from io import BytesIO
from math import ceil
import pandas as pd
import time
import os
from coordinatetransformer import wgs84togcj02, gcj02tobd09ll, bd09lltobd09mc
import urllib.parse
import hashlib

def baidu_sn(query_str, sk):
    """
    生成百度API请求的sn签名
    :param query_str: 不带host的url参数部分，如'/geoconv/v1/?coords=xxx&from=1&to=6&ak=yourak'
    :param sk: 你的百度sk
    :return: sn字符串
    """
    # 1. 对query_str进行URL编码，safe内的保留字符不转换
    encoded_str = urllib.parse.quote(query_str, safe="/:=&?#+!$,;'@()*[]")
    # 2. 拼接sk
    raw_str = encoded_str + sk
    # 3. 用quote_plus再编码一次
    raw_str = urllib.parse.quote_plus(raw_str)
    # 4. md5
    sn = hashlib.md5(raw_str.encode('utf-8')).hexdigest()
    return sn

class BaiduPanoramaSpider:
	"""
	百度全景图片下载器，支持WGS84坐标到BD09MC的本地/在线转换。
	"""
	def __init__(self, ak=None, zoom=3, sleep_sec=3):
		"""
		:param ak: 百度API密钥，若为None或0则使用本地算法
		:param zoom: 图片缩放等级，默认3
		:param sleep_sec: 下载间隔秒数，默认3
		"""
		self.ak = ak
		self.zoom = zoom
		self.sleep_sec = sleep_sec

	def get_image_id(self, x, y):
		"""
		获取指定bd09mc坐标的全景图片ID。
		:param x: bd09mc x
		:param y: bd09mc y
		:return: 图片ID字符串
		"""
		url = f"https://mapsv1.bdimg.com/?qt=qsdata&x={x}&y={y}"
		h = requests.get(url).text
		return json.loads(h)

	def get_image_bytes_list(self, sid, z=None):
		"""
		获取全景图片的分块字节流列表。
		:param sid: 图片ID
		:param z: 缩放等级
		:return: 图片字节流列表
		"""
		if z is None:
			z = self.zoom
		if z == 2:
			xrange, yrange = 1, 2
		elif z == 3:
			xrange, yrange = 2, 4
		elif z == 1:
			xrange, yrange = 1, 1
		elif z == 4:
			xrange, yrange = 4, 8
		img_bytes = []
		for x in range(xrange):
			for y in range(yrange):
				url = f"https://mapsv1.bdimg.com/?qt=pdata&sid={sid}&pos={x}_{y}&z={z}"
				b = requests.get(url).content
				img_bytes.append(b)
		return img_bytes

	@staticmethod
	def bytes_to_img(img_byte):
		"""
		字节流转PIL图片对象。
		:param img_byte: 图片字节流
		:return: PIL.Image
		"""
		return Image.open(BytesIO(img_byte))

	@staticmethod
	def bytes_list_to_img_list(byte_list):
		"""
		字节流列表转图片对象列表。
		:param byte_list: 图片字节流列表
		:return: PIL.Image列表
		"""
		return [BaiduPanoramaSpider.bytes_to_img(b) for b in byte_list]

	@staticmethod
	def merge_image(img_list, img_num_per_row):
		"""
		合并多张图片为一张大图。
		:param img_list: PIL.Image列表
		:param img_num_per_row: 每行图片数
		:return: 合并后的PIL.Image
		"""
		assert isinstance(img_list[0], Image.Image)
		w, h = img_list[0].size
		row_num = ceil(len(img_list) / img_num_per_row)
		width = w * img_num_per_row
		height = h * row_num
		new_img = Image.new("RGB", (width, height))
		for i, img in enumerate(img_list):
			x = i // img_num_per_row
			y = i % img_num_per_row
			new_img.paste(img, (y * w, x * h))
		return new_img

	def download(self, x, y, fp):
		"""
		下载指定bd09mc坐标的全景图片并保存。
		:param x: bd09mc x
		:param y: bd09mc y
		:param fp: 保存路径
		"""
		img_per_row = {1: 1, 2: 2, 3: 4, 4: 8}
		img_id = self.get_image_id(x, y)["content"]['id']
		img_bytes = self.get_image_bytes_list(img_id)
		img_list = self.bytes_list_to_img_list(img_bytes)
		img = self.merge_image(img_list, img_per_row[self.zoom])
		img.save(fp)

	@staticmethod
	def input_points(fp):
		"""
		读取csv文件，按每100个点分组，返回分组字符串列表。
		:param fp: csv文件路径
		:return: ["lon1,lat1;lon2,lat2;...", ...]
		"""
		points = pd.read_csv(fp, encoding="utf8").to_numpy().tolist()
		points100 = []
		for i, (x, y) in enumerate(points):
			n = i // 100
			if len(points100) == n:
				points100 += [f"{x},{y}"]
			else:
				points100[n] += f";{x},{y}"
		return points100

	@staticmethod
	def convert_wgs_to_bd09mc(coords, ak, sk=None):
		"""
		使用百度API将WGS84批量转BD09MC。
		:param coords: 分组字符串
		:param ak: 百度API密钥
		:param sk: 百度安全密钥（可选）
		:return: [[x, y], ...]
		"""
		base_query = f"/geoconv/v1/?coords={coords}&from=1&to=6&ak={ak}"
		if sk:
			sn = baidu_sn(base_query, sk)
			url = f"http://api.map.baidu.com{base_query}&sn={sn}"
		else:
			url = f"http://api.map.baidu.com{base_query}"
		h = requests.get(url)
		points = json.loads(h.text)
		if 'result' not in points:
			raise Exception(f"Baidu API error: {points}")
		return [[x["x"], x["y"]] for x in points['result']]

	@staticmethod
	def convert_wgs_to_bd09mc_without_ak(coords):
		"""
		本地算法将WGS84批量转BD09MC。
		:param coords: 分组字符串列表
		:return: [[x, y], ...]
		"""
		result = []
		for group in coords:
			points = [tuple(map(float, p.split(','))) for p in group.split(';')]
			for lon, lat in points:
				gcj_lon, gcj_lat = wgs84togcj02(lon, lat)
				bd_lon, bd_lat = gcj02tobd09ll(gcj_lon, gcj_lat)
				x, y = bd09lltobd09mc(bd_lon, bd_lat)
				result.append([x, y])
		return result

	def download_from_csv(self, points_csv_path, to_folder_path, sk=None):
		"""
		主入口：根据csv批量下载全景图片。
		:param points_csv_path: 输入csv路径（WGS84坐标）
		:param to_folder_path: 输出图片文件夹
		:param sk: 百度安全密钥（可选）
		"""
		points100 = self.input_points(points_csv_path)
		points = []
		for p in points100:
			if not self.ak or self.ak == 0:
				points += self.convert_wgs_to_bd09mc_without_ak([p])
			else:
				points += self.convert_wgs_to_bd09mc(p, self.ak, sk)
		for i, (x, y) in enumerate(points):
			fp = os.path.join(to_folder_path, f"{i:0>5d}.jpg")
			self.download(x, y, fp)
			time.sleep(self.sleep_sec)
			print(fp)

# 保持兼容原有函数名
def baiduImgDownloader(pointsCsvPath, toFolderPath, ak, zoom=3, sk=None):
	"""
	兼容旧接口，调用类方法。
	:param sk: 百度安全密钥（可选）
	"""
	spider = BaiduPanoramaSpider(ak=ak, zoom=zoom)
	spider.download_from_csv(pointsCsvPath, toFolderPath, sk=sk)

def getImageID(x,y):
	"""
	获取指定坐标的图片ID。
	"""
	url = f"https://mapsv1.bdimg.com/?qt=qsdata&x={x}&y={y}"
	h = requests.get(url).text
	return json.loads(h)

def getImageBytesList(sid,z=2):
	"""
	获取指定图片ID的字节流列表。
	"""
	if z==2:
		xrange,yrange=1,2
	elif z==3:
		xrange,yrange=2,4
	elif z==1:
		xrange,yrange=1,1
	elif z==4:
		xrange,yrange=4,8
	imgBytes=[]
	for x in range(xrange):
		for y in range(yrange):
			url = f"https://mapsv1.bdimg.com/?qt=pdata&sid={sid}&pos={x}_{y}&z={z}"
			b = requests.get(url).content
			imgBytes.append(b)
	return imgBytes

def bytes2Img(imgByte):
	"""
	将字节流转换为PIL.Image。
	"""
	return Image.open(BytesIO(imgByte))

def bytesList2ImgList(x):
	"""
	将字节流列表转换为PIL.Image列表。
	"""
	return [bytes2Img(_) for _ in x]

def mergeImage(imgList,imgNumPerRow):
	"""
	将PIL.Image列表按行合并为一张大图。
	"""
	assert isinstance(imgList[0],Image.Image)
	w,h = imgList[0].size
	rowNum = ceil(len(imgList)/imgNumPerRow)
	width = w * imgNumPerRow
	height = h * rowNum
	newImg = Image.new("RGB",(width,height))
	for i,img in enumerate(imgList):
		x = i//imgNumPerRow
		y = i%imgNumPerRow
		newImg.paste(img,(y*h,x*w,))
	return newImg

def download(x,y,zoom,fp):
	"""
	下载指定坐标的全景图。
	:param x: x坐标 (bd09mc)
	:param y: y坐标 (bd09mc)
	:param zoom: 缩放级别
	:param fp: 保存路径
	"""
	imgPerRow = {1:1,2:2,3:4,4:8}
	imgId = getImageID(x,y)["content"]['id']
	imgBytes = getImageBytesList(imgId,z=zoom)
	imgList = bytesList2ImgList(imgBytes)
	img = mergeImage(imgList,imgPerRow[zoom])
	img.save(fp)
	
def inputPoints(fp):
	"""
	读取CSV文件中的点坐标。
	:param fp: CSV文件路径
	:return: List of str which consists 100 points
	"""
	points = pd.read_csv(fp,encoding="utf8")
	points = points.to_numpy().tolist()
	points100 =[]
	for i,(x,y) in enumerate(points):
		n = i//100
		if len(points100)==n:
			points100 += [f"{x},{y}"]
		else:
			points100[n] += f";{x},{y}"
	return points100	

def convertWGStoBD09MC(coords, ak, sk=None):
	"""
	将WGS84坐标转换为BD09MC坐标。
	:param coords: WGS84坐标列表
	:param ak: 百度API密钥
	:param sk: 百度安全密钥（可选）
	:return: BD09MC坐标列表
	"""
	base_query = f"/geoconv/v1/?coords={coords}&from=1&to=6&ak={ak}"
	if sk:
		sn = baidu_sn(base_query, sk)
		url = f"http://api.map.baidu.com{base_query}&sn={sn}"
	else:
		url = f"http://api.map.baidu.com{base_query}"
	h = requests.get(url)
	points = json.loads(h.text)
	# print(points)
	if 'result' not in points:
		raise Exception(f"Baidu API error: {points}")
	points = [[x["x"], x["y"]] for x in points['result']]
	# 打印坐标转换结果
	for p in points:
		print(f"WGS84: ({p[0]}, {p[1]}) -> BD09MC: ({p[0]}, {p[1]})")
	return points

def convertWGStoBD09MCwithoutak(coords):
	"""
	将WGS84坐标转换为BD09MC坐标。
	:param coords: WGS84坐标列表
	:return: BD09MC坐标列表
	"""
	result = []
	for group in coords:
		points = [tuple(map(float, p.split(','))) for p in group.split(';')]
		for lon, lat in points:
			gcj_lon, gcj_lat = wgs84togcj02(lon, lat)
			bd_lon, bd_lat = gcj02tobd09ll(gcj_lon, gcj_lat)
			x, y = bd09lltobd09mc(bd_lon, bd_lat)
			result.append([x, y])
			print(f"WGS84: ({lon}, {lat}) -> BD09MC: ({x}, {y})")
	return result

def baiduImgDownloader(pointsCsvPath, toFolderPath, ak, zoom=3, sk=None):
	"""
	下载百度全景图。
	:param pointsCsvPath: 输入的CSV文件路径
	:param toFolderPath: 输出的文件夹路径
	:param ak: 百度API密钥
	:param zoom: 缩放级别
	:param sk: 百度安全密钥（可选）
	"""
	points100 = inputPoints(pointsCsvPath)
	points = []
	for p in points100:
		if ak == 0:
			points += convertWGStoBD09MCwithoutak([p])
		else:
			points += convertWGStoBD09MC(p, ak, sk)
	for i, (x, y) in enumerate(points):
		fp = os.path.join(toFolderPath, f"{i:0>5d}.jpg")
		download(x, y, zoom, fp)
		time.sleep(3)
		print(fp)

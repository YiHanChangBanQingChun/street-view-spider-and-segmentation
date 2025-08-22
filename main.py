from imgspider import baiduImgDownloader

YourAK = "YourAK"
YourSK = "YourSK"


# 使用ak与sk
baiduImgDownloader("resources/example.csv", # CRS: WGS84
					 "resources/downloadPic", # folder
					 ak=YourAK, zoom=3, sk=YourSK)
# 不使用ak与sk
baiduImgDownloader("resources/example.csv", # CRS: WGS84
					"resources/downloadPic", # folder
					ak=0, zoom=3, sk=None)
import cv2
import numpy as np
import os
import exifread
import math
import xlrd
import xlwt
import math
import matplotlib.pyplot as plot


# matplotlib.get_cachedir()
def convert_to_decimal(*gps):
    # 度
    if '/' in gps[0]:
        deg = gps[0].split('/')
        if deg[0] == '0' or deg[1] == '0':
            gps_d = 0
        else:
            gps_d = float(deg[0]) / float(deg[1])
    else:
        gps_d = float(gps[0])
    # 分
    if '/' in gps[1]:
        minu = gps[1].split('/')
        if minu[0] == '0' or minu[1] == '0':
            gps_m = 0
        else:
            gps_m = (float(minu[0]) / float(minu[1])) / 60
    else:
        gps_m = float(gps[1]) / 60
    # 秒
    if '/' in gps[2]:
        sec = gps[2].split('/')
        if sec[0] == '0' or sec[1] == '0':
            gps_s = 0
        else:
            gps_s = (float(sec[0]) / float(sec[1])) / 3600
    else:
        gps_s = float(gps[2]) / 3600

    decimal_gps = gps_d + gps_m + gps_s
    # 如果是南半球或是西半球
    if gps[3] == 'W' or gps[3] == 'S' or gps[3] == "83" or gps[3] == "87":
        return str(decimal_gps * -1)
    else:
        return str(decimal_gps)
from matplotlib import pyplot as plt
def imageread(path):
    f = open(path, 'rb')
    GPS = {}
    Data = ""
    try:
        tags = exifread.process_file(f)
    except:
        return
    #print(tags)
    '''
    for tag in tags:
        print(tag,":",tags[tag])
    '''

    # 南北半球标识
    if 'GPS GPSLatitudeRef' in tags:

        GPS['GPSLatitudeRef'] = str(tags['GPS GPSLatitudeRef'])
        # print(GPS['GPSLatitudeRef'])
    else:
        GPS['GPSLatitudeRef'] = 'N'  # 缺省设置为北半球

    # 东西半球标识
    if 'GPS GPSLongitudeRef' in tags:
        GPS['GPSLongitudeRef'] = str(tags['GPS GPSLongitudeRef'])
        # print(GPS['GPSLongitudeRef'])
    else:
        GPS['GPSLongitudeRef'] = 'E'  # 缺省设置为东半球

    # 海拔高度标识
    if 'GPS GPSAltitudeRef' in tags:
        GPS['GPSAltitudeRef'] = str(tags['GPS GPSAltitudeRef'])
        #print(GPS['GPSAltitudeRef'])
    # 获取纬度
    if 'GPS GPSLatitude' in tags:
        lat = str(tags['GPS GPSLatitude'])
        # 处理无效值
        if lat == '[0, 0, 0]' or lat == '[0/0, 0/0, 0/0]':
            return

        deg, minu, sec = [x.replace(' ', '') for x in lat[1:-1].split(',')]
        # 将纬度转换为小数形式
        GPS['GPSLatitude'] = convert_to_decimal(deg, minu, sec, GPS['GPSLatitudeRef'])

    # 获取经度
    if 'GPS GPSLongitude' in tags:
        lng = str(tags['GPS GPSLongitude'])
        #print(lng)

        # 处理无效值
        if lng == '[0, 0, 0]' or lng == '[0/0, 0/0, 0/0]':
            return

        deg, minu, sec = [x.replace(' ', '') for x in lng[1:-1].split(',')]
        # 将经度转换为小数形式
        GPS['GPSLongitude'] = convert_to_decimal(deg, minu, sec, GPS['GPSLongitudeRef'])  # 对特殊的经纬度格式进行处理

    # 获取海拔高度
    if 'GPS GPSAltitude' in tags:
        GPS['GPSAltitude'] = str(tags["GPS GPSAltitude"])
        #print(GPS['GPSAltitude'] )

    # 获取图片拍摄时间
    if 'Image DateTime' in tags:
        GPS["DateTime"] = str(tags["Image DateTime"])
        #print(GPS["DateTime"])
    elif "EXIF DateTimeOriginal" in tags:
        GPS["DateTime"] = str(tags["EXIF DateTimeOriginal"])

    return GPS['GPSLatitude'], GPS['GPSLongitude']


#计算目标组串与中心点的距离和方位角
def calc_angle(x2,y2,x1,y1,scale):
    a=x1-x2
    b=y1-y2
    m=math.sqrt((a**2)+(b**2))
    # M=m*26.8 #红外像素和真实尺寸比例比例
    # #m×3.3 可见光比例
    M=m*scale
    angle=0
    dy= y2-y1
    dx= x2-x1
    print("dx",dx)
    print("dy",dy)
    if dx == 0 and dy > 0:
        angle = 0
    if dx == 0 and dy < 0:
        angle = 180
    if dy == 0 and dx > 0:
        angle = 90
    if dy == 0 and dx < 0:
        angle = 270
    if dx > 0 and dy > 0:
        angle = math.atan(dx / dy) * 180 / math.pi
    elif dx < 0 and dy > 0:
        angle = 360 + math.atan(
            dx / dy) * 180 / math.pi
    elif dx < 0 and dy < 0:
        angle = 180 + math.atan(dx / dy) * 180 / math.pi
    elif dx > 0 and dy < 0:
        angle = 180 + math.atan(dx / dy) * 180 / math.pi
    return angle,M

#计算目标经纬度
#gps:原点经纬度, ang:目标方位角（角度）， dist:目标距离(米)；
def GPScalculate(gps,ang,dist):
    lat = gps[0]  # 原点纬度
    lon = gps[1]  # 原点经度

    lat1 = float(lat) / 180 * math.pi
    lon1 = float(lon) / 180 * math.pi
    # lat1 = lat * math.pi / 180
    # lon1 = lon * math.pi / 180
    brg = ang * math.pi / 180
    #print(brg)
    #地球扁率
    flat = 298.257223563
    #地球半长轴
    a = 6378137.0
    #地球半短轴
    b = 6356752.314245
    f = 1 / flat
    sb = math.sin(brg)
    cb = math.cos(brg)
    tu1 = (1 - f) * math.tan(lat1)
    cu1 = 1 / math.sqrt((1 + tu1 * tu1))
    su1 = tu1 * cu1
    s2 = math.atan2(tu1, cb)
    sa = cu1 * sb
    csa = 1 - sa * sa
    us = csa * (a * a - b * b) / (b * b)
    A = 1 + us / 16384 * (4096 + us * (-768 + us * (320 - 175 * us)))
    B = us / 1024 * (256 + us * (-128 + us * (74 - 47 * us)))
    s1 = dist / (b * A)
    s1p = 2 * math.pi
    cs1m = 0.0
    ss1 = 0.0
    cs1 = 0.0
    ds1 = 0.0

    while abs(s1 - s1p) > 1e-12:
        cs1m = math.cos(2 * s2 + s1)
        ss1 = math.sin(s1)
        cs1 = math.cos(s1)
        ds1 = B * ss1 * (cs1m + B / 4 * (
                cs1 * (-1 + 2 * cs1m * cs1m) - B / 6 * cs1m * (-3 + 4 * ss1 * ss1) * (-3 + 4 * cs1m * cs1m)))
        s1p = s1
        s1 = dist / (b * A) + ds1
    t = su1 * ss1 - cu1 * cs1 * cb
    lat2 = math.atan2(su1 * cs1 + cu1 * ss1 * cb, (1 - f) * math.sqrt(sa * sa + t * t))
    l2 = math.atan2(ss1 * sb, cu1 * cs1 - su1 * ss1 * cb)
    c = f / 16 * csa * (4 + f * (4 - 3 * csa))
    l = l2 - (1 - c) * f * sa * (s1 + c * ss1 * (cs1m + c * cs1 * (-1 + 2 * cs1m * cs1m)))
    lon2 = lon1 + l

    print("目标纬度：", lat2 * 180 / math.pi, "  目标经度：", lon2 * 180 / math.pi)
    return (lon2 * 180 / math.pi, lat2 * 180 / math.pi)


def millerToXY (lon, lat):
    """
    经纬度转换为平面坐标    #('30.192849666666667', '120.1979956388889')
    #('30.19284961111111', '120.19799491666667')
    #('30.192848916666666', '120.19799475')系中的x,y 利用米勒坐标系
    :param lon: 经度
    :param lat: 维度
    :return:
    """
    xy_coordinate = []
    L = 6381372*math.pi*2
    W = L
    H = L/2
    mill = 2.3
    x = float(lon)*math.pi/180
    y = float(lat)*math.pi/180
    y = 1.25*math.log(math.tan(0.25*math.pi+0.4*y))
    x = (W/2)+(W/(2*math.pi))*x
    y = (H/2)-(H/(2*mill))*y
    xy_coordinate.append((int(round(x)),int(round(y))))
    #return xy_coordinate
    return x,y

def cordtoxy(filepath,areas):
    #该函数读取各个区域组串坐标及GPS信息的excel，转化为组串号并存入新的excel文件，为后续组件缺陷识别所用

    myWorkbook = xlrd.open_workbook(filepath)
    mySheets = myWorkbook.sheets()
    mySheet = mySheets[0]
    x=[]
    y=[]
    x=mySheet.col_values(0)
    y=mySheet.col_values(1)
    log=mySheet.col_values(2)
    lat=mySheet.col_values(3)
    xy={}
    for i in range(len(y)):
        if y[i] not in xy:
            xy[y[i]]=[]
        xy[y[i]].append(x[i])
    #print('zx',xy)
    zx = sorted(xy.items(), key=lambda x: x[0])
    #print('zx',zx)
    X=[]
    Y=[]
    for x2,y2 in xy.items():
        X.append(x2)
        Y.append(y2)
    j=[]
    k=[]
    m=[]
    xsorted=[]
    ysorted=[]
    for i in range(len(zx)):
        if len(zx[i][1]) == 1:
            j=(zx[i][0],zx[i][1][0])
            m.append(j)
        elif len(zx[i][1]) != 1:
            for v in range(len(zx[i][1])):

                j = (zx[i][0], zx[i][1][v])
                m.append(j)
        xsorted.append(zx[i][0])
        ysorted.append(zx[i][1])

    #print("m",m)
    #ysorted.sort()
    jj=[]
    kk=[]
    n=0
    N=[]
    for i in range(len(ysorted)):
         #print(xsorted[i],ysorted[i])

         if i+1==len(zx):

             jj.extend(ysorted[i])
             kk.append(jj)
             break
         if abs(int(xsorted[i+1])-int(xsorted[n]))<=3:
            jj.extend(ysorted[i])
         else:
             N.append(n)
             n=i+1
             jj.extend(ysorted[i])
             kk.append(jj)
             jj=[]
    #print('N',N)

    for i in range(len(kk)):
        kk[i] = list(map(int, kk[i]))
        kk[i].sort()
    #print(kk)

    #result为所有组串组成的行列
    result=[[(i,j) for j in range(len(kk[i]))] for i in range(len(kk))]
    #print(result)

    #新的excel文件，为组串号信息
    res_workbook = xlwt.Workbook()
    res_worksheet = res_workbook.add_sheet(u'My Worksheet', cell_overwrite_ok=True)
    t=0
    #print('xso',xsorted)
    #print('yso',ysorted)
    for i in range(len(ysorted)):
        ysorted[i].sort()
    #print('ysoed',ysorted)
    mm=[]
    nn=[]
    b = 0
    for i in range(len(kk)):

        for j in range(len(kk[i])):

            c=j+b
            nn.append(c)
        
        mm.append(nn)
        nn=[]
        b=len(kk[i])+b
    # print('mm',mm)
    # print(m)
    qq=[]
    pp=[]
    for j in range(len(mm)):
        for k in range(len(mm[j])):
            pp.append(m[mm[j][k]])

        qq.append(pp)
        pp = []
    #print('qq',qq)
    for i in range(len(qq)):
        qq[i]=sorted(qq[i], key=lambda s:s[1])
    #print('sorted qq',qq)

    newm=[]
    for i in range (len(qq)):
        for j in range(len(qq[i])):
            newm.append(qq[i][j])

    #print('newm',newm)
    #print(len(newm))

    for i in range(len(newm)):
        res_worksheet.write(t, 0, str(int(newm[i][1]))+","+str(int(newm[i][0])))
        t+=1
        res_worksheet.write(newm.index((int(y[i]), float(x[i]))), 1, float(log[i]))
        res_worksheet.write(newm.index((int(y[i]), float(x[i]))), 2, float(lat[i]))

    t=0
    for i in range(len(result)):
        for j in range(0,len(result[i])):
            res_worksheet.write(t, 3, 'A'+str(areas)+'H'+str(result[i][j][0]+1)+'L'+str(result[i][j][1]+1))
            t+=1

    res_workbook.save('区域'+str(areas)+'坐标.xls')

    return result

def Gray_img(src):
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    #cv2.namedWindow('gray', cv2.WINDOW_NORMAL)
    #cv2.imshow('gray', gray)
    #cv2.imwrite('gray.png', gray)

    return gray

def Exteact_temp(img):
    gray_img = Gray_img(img)
    blur = cv2.GaussianBlur(gray_img, (5, 5), 0)
    blurshape=blur.shape
    #grayvalue=0
    grayvaluemax=0
    for i in range(blurshape[0]):
        for j in range(blurshape[1]):
            grayvalue = blur[i,j]#求灰度值
            if grayvalue >grayvaluemax:
                grayvaluemax=grayvalue
    #print(grayvaluemax)
    a=int(grayvaluemax)*int(grayvaluemax)*0.0021
    #print(a)
    b=-0.35*grayvaluemax
    c=40.53
    temp=a+b+c
    return temp

def BinGrid(sizes,areas):
    # # 饼状图
    # plot.figure(figsize=(8,8))


    allboard=sizes[0]+sizes[1]+sizes[2]+sizes[3]
    badboard=allboard-sizes[0]
    plot.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    badsize=sizes[1:4]
    labels = ['热斑: '+str(badsize[0]),  '二极管故障: '+str(badsize[1]), '阴影遮挡 :'+str(badsize[2])]

    plot.figure()
    patches, l_text, p_text=0,0,0
    patches, l_text, p_text = plot.pie(badsize, labels=labels,
                                       labeldistance=1.1, autopct='%2.0f%%', shadow=False,
                                       startangle=90, pctdistance=0.6)

    for t in l_text:
        t.set_size = 30
    for t in p_text:
        t.set_size = 20
    # 设置x，y轴刻度一致，这样饼图才能是圆的
    plot.text(0, -1.2, '区域'+str(areas)+' 全部组件:'+str(allboard)+'; 缺陷组件：'+str(badboard))
    plot.axis('equal')
    plot.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))

    plot.grid()
    plot.savefig('./'+'缺陷统计'+str(areas)+'.jpg')

    #plot.show()

def Hot_BinGrid(sizes,areas):
    plot.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签


    labels = ['轻度热斑: '+str(sizes[0]),  '中度热斑: '+str(sizes[1]), '重度热斑 :'+str(sizes[2])]

    plot.figure()
    patches, l_text, p_text=0,0,0
    patches, l_text, p_text = plot.pie(sizes, labels=labels,
                                       labeldistance=1.1, autopct='%2.0f%%', shadow=False,
                                       startangle=90, pctdistance=0.6)

    for t in l_text:
        t.set_size = 30
    for t in p_text:
        t.set_size = 20
    # 设置x，y轴刻度一致，这样饼图才能是圆的
    plot.text(0, -1.2, '区域'+str(areas))
    plot.axis('equal')
    plot.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))

    plot.grid()
    plot.savefig('./'+'热斑性质统计'+str(areas)+'.jpg')

def all_error_board_Bingrid(allsize):
    plot.figure()
    allboard=np.sum(allsize)
    allerrorboard=allboard-allsize[0]
    labels = ['全部正常组件: '+str(allsize[0]),'全部热斑组件: '+str(allsize[1]),'全部二极管故障组件: '+str(allsize[2]),'全部阴影遮挡组件: '+str(allsize[3])]
    patches, l_text, p_text=0,0,0

    patches, l_text, p_text = plot.pie(allsize, labels=labels,
                                       labeldistance=1.1, autopct='%2.0f%%', shadow=False,
                                       startangle=90, pctdistance=0.6)
    for t in l_text:
        t.set_size = 30
    for t in p_text:
        t.set_size = 20
        # 设置x，y轴刻度一致，这样饼图才能是圆的
    plot.text(0, -1.2, '全电站所有组件:' + str(allboard) + '; 全部缺陷组件：' + str(allerrorboard))
    plot.axis('equal')
    plot.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))

    plot.grid()
    plot.savefig('all.jpg')
def all_Hot_BinGrid(sizes):
    plot.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签


    labels = ['所有轻度热斑: '+str(sizes[0]),  '所有中度热斑: '+str(sizes[1]), '所有重度热斑 :'+str(sizes[2])]

    plot.figure()
    patches, l_text, p_text=0,0,0
    patches, l_text, p_text = plot.pie(sizes, labels=labels,
                                       labeldistance=1.1, autopct='%2.0f%%', shadow=False,
                                       startangle=90, pctdistance=0.6)

    for t in l_text:
        t.set_size = 30
    for t in p_text:
        t.set_size = 20
    # 设置x，y轴刻度一致，这样饼图才能是圆的
    #plot.text(0, -1.2, '区域'+str(areas))
    plot.axis('equal')
    plot.legend(loc='upper left', bbox_to_anchor=(-0.1, 1))

    plot.grid()
    plot.savefig('./'+'全电站热斑性质统计'+'.jpg')

#求像素和真实尺寸比例
# f=h*D/H
# f=焦距 h=CCD的对角线尺寸 D=物体到镜头的距离 H=图片实际对角线距离（单位同D）
# 知道H后，图片的对角线像素尺寸 除以 H, 可以得到每个像素和真实距离的比例Z
# x乘以Z就是物体实际大小（单位同D）
def cal_ratio(ratio,equal_ratio,drone_height,height,width):
    ccd=43.27  #35mm传感器对角线，不用改

    #图像对角线
    img_scale=math.sqrt(math.pow(height,2)+math.pow(width,2))

    #实际焦距
    ratio_value=ratio*ccd/equal_ratio

    #实际图像对角线距离
    img_value=drone_height*ratio_value/ratio

    #真实图像与像素比例
    scale=img_value*1000/img_scale

    return scale





if __name__ == '__main__':

    result = cordtoxy('组串坐标.xls')

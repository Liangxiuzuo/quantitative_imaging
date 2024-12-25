import numpy as np
import pandas as pd
import numpy as np
import csv
import xlrd
import math
'''
def make_bg_pdf(background_file="background-camera.csv"):
    ##每个像素的低能阈值不同，可以已知本底能谱做出每个像素的本底计数响应。
    #step.1
    import numpy as np
    np.random.seed(1)
    pixel_total = 361;
    cut_ene = 50;#keV
    sigma_ene = 10;#keV
    np.random.seed(0)
    pixel_all_cut_ene = np.random.normal(cut_ene, sigma_ene,pixel_total);
    Column_index=['Counts']
    background_spec=pd.read_csv(background_file,",",header=None,names=Column_index,index_col=False)
    #step.2
    pdfbg=np.zeros((361))#根据每个像素的能量低阈，并且参考本底能谱，设置每个像素的本底计数相对响应概率：阈值越低，获得本底计数越多。
    for i in range(361):
        cut_ene=int(pixel_all_cut_ene[i]);
        pdfbg[i]=np.sum(background_spec[cut_ene:2048]);
'''

def recall():
    print('test；',np.random.randint(1,19));



## 计算361像素的抽样投影
def make_mask_projection(mask_pattern_pdf,fornum):
#def make_background_projection(mask_pdf,fornum):
    np.random.seed(1)
    ##每个像素的低能阈值不同，可以已知本底能谱做出每个像素的本底计数响应。
    #step.1
    np.random.seed(1)
    mask_projection=np.zeros((361));
    pro_index=np.random.choice(range(361),size=(fornum),p=mask_pattern_pdf.ravel());
    for i in range(361):
        mask_projection[i]=np.sum(pro_index==i);
    return mask_projection.reshape(19,19);
'''
 def make_bg_frompdf(bg_pattern_pdf,fornum):
#def make_background_projection(mask_pdf,fornum):
    np.random.seed(1)
    ##每个像素的低能阈值不同，可以已知本底能谱做出每个像素的本底计数响应。
    #step.1
    np.random.seed(1)
    mask_projection=np.zeros((361));
    pro_index=np.random.choice(range(361),size=(fornum),p=mask_pattern_pdf.ravel());
    for i in range(361):
        mask_projection[i]=np.sum(pro_index==i);
    return mask_projection.reshape(19,19);
'''   

## 计算361像素的本底噪声
def make_background_projection(background_file,fornum):
#def make_background_projection(background_file="background-camera.csv",fornum):
    np.random.seed(1)
    ##每个像素的低能阈值不同，可以已知本底能谱做出每个像素的本底计数响应。
    #step.1
    np.random.seed(1)
    pixel_total = 361;
    cut_ene = 50;#keV
    sigma_ene = 10;#keV
    np.random.seed(0)
    pixel_all_cut_ene = np.random.normal(cut_ene, sigma_ene,pixel_total);
    Column_index=['Counts']
    background_spec=pd.read_csv(background_file,",",header=None,names=Column_index,index_col=False)
    #step.2
    pdfbg=np.zeros((361))#根据每个像素的能量低阈，并且参考本底能谱，设置每个像素的本底计数相对响应概率：阈值越低，获得本底计数越多。
    for i in range(361):
        cut_ene=int(pixel_all_cut_ene[i]);
        pdfbg[i]=np.sum(background_spec[cut_ene:2048])*100;## 概率值为整数

## rate is the pdf, fornum is the total number for distribution
    #fornum=
    bg_outcome=np.zeros((361))
    for i in range(fornum):
        start = 0
        index = 0
        randnum = np.random.randint(1, sum(pdfbg))
    
        for index, scope in enumerate(pdfbg):
            start += scope
            if randnum <= start:
                break    
        bg_outcome[index]= bg_outcome[index]+1
    #    return outcome
    # ooooooo up is the 
    #bg_camera_pos_360=randomlxzpdf(pdfbg*100,40*360);#本底计数：40/s，sample time:360
    pixel_all_bg_positive_360=np.zeros((19,19));# 每个像素的低能阈值不同
    for i in range(19):
        for j in range(19):
            pixel_all_bg_positive_360[i][j]=bg_outcome[i*19+j];
    return        pixel_all_bg_positive_360;



def geant4_projection_from_file(file_name,start_num,event_num,mark_mask):
    ## event_num is equal to the mean cps by time
    Column_index=['event_id','ene','pixel_x','pixel_y','mask_angle','source_theta','source_phi','source_energy'];
    sim_data=pd.read_csv(file_name,",",header=None,names=Column_index,index_col=False)
    #sim_data=pd.read_csv("camera_mura19_0510.csv",",",header=None,names=Column_index,index_col=False)
#    return  sim_data_positive;
#    time_num=int(time_use*len(sim_data));
    g4_pro=np.zeros((19,19));
    for i in range(start_num,event_num+start_num):
#        if sim_data['ene'][i]*1000>pixel_cut_ene[sim_data['pixel_y'][i]][sim_data['pixel_x'][i]]:
        #if positive_sim_data['ene'][i]>0.05:
  
        g4_pro[sim_data['pixel_y'][i]][sim_data['pixel_x'][i]]+=1;
        #g4_pro[sim_data['pixel_y'][i]][sim_data['pixel_x'][i]]=g4_pro[sim_data['pixel_y'][i]][sim_data['pixel_x'][i]]+1;
    return g4_pro;
    #return g4_pro,sim_data;

def geant4_projection_of(sim_data_frame,event_num):
    ## event_num is equal to the mean cps by time
    Column_index=['event_id','ene','pixel_x','pixel_y'];
    g4_pro=np.zeros((19,19));
    for i in range(event_num):
#        if sim_data['ene'][i]*1000>pixel_cut_ene[sim_data['pixel_y'][i]][sim_data['pixel_x'][i]]:
        #if positive_sim_data['ene'][i]>0.05:
  
        g4_pro[sim_data_frame['pixel_y'][i]][sim_data_frame['pixel_x'][i]]+=1;
    return g4_pro;




def make_final_projection(g4_data,backround): 
    # time_use: 1 means two time periods, 0.5 is one time period
       #if positive_sim_data['ene'][i]>0.05:
  
    return g4_data+backround
        
    
def decode_mura19(pro_data,mark_mask,non_negative):
    #mark_mask=1;
    positive_uncode=np.array([
        [1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1]]);
    negative_uncode=np.array([
        [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
        [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1]]);
    if mark_mask==1:
        uncode=positive_uncode;
    else:
        uncode=negative_uncode;
    rows,columns=pro_data.shape[0],pro_data.shape[1];

    decode=np.zeros((rows,columns));
    for i in range(rows):
        for j in range(columns):
            for k in range(rows):
                for m in range(columns):
                    decode[i][j]=decode[i][j]+uncode[k][m]*pro_data[(i+k)%rows][(j+m)%columns];

                            #print('rows: ',rows)

    if non_negative>0.5:
        decode[decode<0.]=np.int(0); # 取非负，符合物理意义                      
    return decode


def calsnr(ima):
    '''
    ima should be a np.array(ima);
    SNR is the peak-snr, peak value/ 本底标准差
    
    '''
    total_counts=ima.sum(); peak_val=ima.max();peak_to_all_ratio=peak_val/total_counts;
    ima_max=ima.max();
    ima_modi=np.delete(ima,ima.argmax(), None);
    snr=(ima_max-ima_modi.mean())/ima_modi.std();
#
#
    print('SNR is: ',snr,' total counts:',total_counts,' peak value:',peak_val,' peak ratio:',peak_to_all_ratio);
    return snr
def calsnrofhot(ima):


    rows,columns=ima.shape[0],ima.shape[1];
    ima_max=ima.max();
    max_position=np.where(ima==ima_max);
    max_row=max_position[0][0];max_column=max_position[1][0];
    hot_val_sum=0;
    hot_area_index=np.zeros(9);

    for i in range(3):
        for j in range(3):
            hot_val_sum+=ima[(i-1+max_row+rows)%rows,(j-1+max_column)%columns];
            hot_area_index[i*3+j]=int(columns*((i-1+max_row+rows)%rows)+(j-1+max_column)%columns);
            #print('index:',int(hot_area_index[i*3+j]));
    hot_area_index_int=hot_area_index.astype(int);
    bg_mean=(ima.sum()-hot_val_sum)/(rows*columns-9);
    bg_array=np.delete(ima,hot_area_index_int);
    hot_val=(hot_val_sum-9*bg_array.mean());
    print('9 times bg pixels:',9*bg_array.mean());
    hot_snr=(ima_max-bg_array.mean())/bg_array.std();
    #hot_snr=hot_val/bg_array.std();
    print('Hot spot information of hot_snr,clean_hot_val,hot_val_sum of 9 pixels, image peak, 3*bg std:',round(hot_snr,3),',',round(hot_val,3),',',round(hot_val_sum,3),',',round(ima_max,3),',',round(3*bg_array.std()));
    return [round(hot_snr,3), round(hot_val,3),round(hot_val_sum,3),round(ima_max,3),round(3*bg_array.std(),3)];




def getCamera_nSv(in_ene):
    ene_term=201;
    ref_nSv_ene=np.arange(201)*0.01;

    ref_nSv=np.array([ 0.  ,0.14223 ,0.0689137 ,0.0908994 ,0.11021 ,0.136743 ,0.163596 ,0.188256 ,0.214878 ,0.240873 ,0.31763 ,0.339704 ,0.358571 ,0.374114 ,0.393888 ,0.413658 ,0.467228 ,0.491125 ,0.507365 ,0.537029 ,0.564554 ,0.592699 ,0.626983 ,0.658857 ,0.69067 ,0.72623 ,0.764354 ,0.803867 ,0.846834 ,0.90301 ,0.941094 ,0.976691 ,1.03134 ,1.07307 ,1.12343 ,1.16515 ,1.23345 ,1.28462 ,1.33348 ,1.39814 ,1.44207 ,1.50224 ,1.56446 ,1.61013 ,1.68861 ,1.73427 ,1.82321 ,1.88026 ,1.94405 ,2.02823 ,2.09101 ,2.16929 ,2.21806 ,2.27306 ,2.37858 ,2.42375 ,2.51829 ,2.5971 ,2.71927 ,2.79727 ,2.83558 ,2.90269 ,2.98896 ,3.08112 ,3.14398 ,3.22086 ,3.32175 ,3.38206 ,3.44201 ,3.55069 ,3.64052 ,3.79168 ,3.83361 ,3.9553 ,4.01166 ,4.1047 ,4.26973 ,4.23435 ,4.36207 ,4.51122 ,4.52927 ,4.62546 ,4.74029 ,4.84963 ,4.94028 ,4.99963 ,5.09869 ,5.1613 ,5.37203 ,5.35541 ,5.43786 ,5.56106 ,5.69776 ,5.72479 ,5.90781 ,5.97673 ,5.92956 ,6.12346 ,6.22226 ,6.30681 ,6.37026 ,6.48554 ,6.60633 ,6.60414 ,6.58262 ,6.71003 ,6.88931 ,6.88735 ,6.99652 ,6.99821 ,7.20301 ,7.2891 ,7.33636 ,7.55251 ,7.32982 ,7.55632 ,7.82986 ,7.61007 ,7.79947 ,7.73951 ,8.21219 ,7.83222 ,8.00428 ,8.16958 ,8.25716 ,8.3522 ,8.44671 ,8.46954 ,8.46128 ,8.54214 ,8.86899 ,8.84707 ,8.90813 ,8.96299 ,8.96124 ,9.03429 ,9.23643 ,9.08087 ,9.25286 ,9.38446 ,9.49238 ,9.66472 ,9.43711 ,9.82596 ,9.98204 ,10.1097 ,9.90515 ,9.92053 ,9.92824 ,10.196 ,9.97826 ,10.5335 ,10.3999 ,10.2803 ,10.644 ,10.709 ,10.5159 ,10.8354 ,10.4656 ,10.9567 ,11.1985 ,10.9576 ,10.9518 ,11.0861 ,11.1633 ,11.3623 ,11.2325 ,11.1916 ,11.3763 ,11.3367 ,11.1008 ,11.4294 ,11.8564 ,11.6347 ,11.6245 ,11.7062 ,11.7468 ,12.2071 ,11.6126 ,12.4222 ,12.2881 ,11.9713 ,12.2534 ,12.0446 ,12.2282 ,11.866 ,12.2413 ,12.633 ,12.4626 ,12.3156 ,13.1471 ,12.3969 ,13.1748 ,12.6177 ,12.6699 ,12.5022 ,12.2159 ,12.629 ,12.8126 ,13.6702 ,12.7383]);


    nSv_val=0;
    for ene_id in range(ene_term):
        if in_ene>=ref_nSv_ene[ene_id] and in_ene<=ref_nSv_ene[ene_id+1]:
            nSv_val=(ref_nSv[ene_id+1]-ref_nSv[ene_id])/(ref_nSv_ene[ene_id+1]-ref_nSv_ene[ene_id])*(in_ene-ref_nSv_ene[ene_id])+ref_nSv[ene_id];
            break;
        
    
    return nSv_val;#unit:nSv/h
   
def testfunc(ene):

    ref_nSv_ene=np.arange(201)*0.01;
    return ref_nSv_ene[3];

'''
                            codearray=np.array([
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0,1,1,0,0,1,1,1,1,0,1,0,1,0,0,0,0,1,1,0],
                                [0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1,1,0,1,1,0,0,0,0,1,0,1,0,1,1,1,1,0,0,1]]);
                            uncode=np.array([
                                [1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1],
                                [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
                                [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
                                [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
                                [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
                                [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
                                [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
                                [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
                                [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
                                [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
                                [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
                                [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
                                [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
                                [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
                                [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
                                [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1],
                                [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
                                [1,1,-1,-1,1,1,1,1,-1,1,-1,1,-1,-1,-1,-1,1,1,-1],
                                [1,-1,1,1,-1,-1,-1,-1,1,-1,1,-1,1,1,1,1,-1,-1,1]]);
                            return i
'''
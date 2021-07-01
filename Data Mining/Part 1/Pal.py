import pandas as pd
from datetime import datetime as dt

def func(df,c1):
    
    cgmAutoDatewise=df.groupby('Date')
    #countperdayAuto=cgmAutoDatewise.size().reset_index()
    #countperdayAuto=countperdayAuto.set_index('Date')
    #print(countperdayAuto)
    #c1=df.loc[df.iloc[:,2]>threshold]      #CGM>180
    c1Datewise=c1.groupby('Date').size().reset_index()
    c1Datewise=c1Datewise.set_index('Date')
    c1Datewise.columns=['Value']
    #print(c1Datewise)
    
    #mergeDatewise=countperdayAuto[countperdayAuto.index.isin(c1Datewise.index)]  #Datewise merge for cgm>180
    #mergeDatewise.columns=['Value']
    percentDatewise=c1Datewise['Value']/288            #CGM>180 count DIVIDED BY total CGM count, PER DAY
    percentDatewise=percentDatewise.to_frame()
    percentDatewise['Value']=percentDatewise['Value']*100
    #print(percentDatewise)
    avg=(percentDatewise['Value'].sum())/len(cgmAutoDatewise)
    #print(avg)
    return avg


cgm1 = pd.read_csv('CGMData.csv',dtype='unicode')
insulin1 = pd.read_csv('InsulinData.csv',dtype='unicode')

cgm = pd.DataFrame(cgm1, columns=['Date','Time','Sensor Glucose (mg/dL)'])
insulin = pd.DataFrame(insulin1, columns=['Date','Time','Alarm'])

cgm['Date']=pd.to_datetime(cgm['Date']).dt.date
cgm['Time']=pd.to_datetime(cgm['Time']).dt.time

cgm['Sensor Glucose (mg/dL)']=pd.to_numeric(cgm['Sensor Glucose (mg/dL)'])

insulin['Date']=pd.to_datetime(insulin['Date']).dt.date
insulin['Time']=pd.to_datetime(insulin['Time']).dt.time

automode=insulin.loc[insulin['Alarm']=='AUTO MODE ACTIVE PLGM OFF']
insulinautodate=automode.iloc[0,0]
insulinautotime=automode.iloc[0,1]

cgmautodate=insulinautodate
cgmauto=cgm.loc[cgm['Date']==insulinautodate]
cgmauto=cgmauto.loc[cgmauto['Time']>insulinautotime]
cgmautotime=cgmauto.iloc[-1,1]

cgmManual=cgm.loc[(cgm['Date']<cgmautodate) | ((cgm['Date']==cgmautodate) & (cgm['Time']<cgmautotime) )]
cgmAuto=cgm.loc[(cgm['Date']>cgmautodate) | ((cgm['Date']==cgmautodate) & (cgm['Time']>=cgmautotime) )]
cgmManual=cgmManual.dropna()
cgmAuto=cgmAuto.dropna()

#--------------------------------------------------------------------------------------------
#WHOLE DAY

#MANUAL
#cgmManual.to_csv('Manual.csv')
c1=cgmManual.loc[cgmManual.iloc[:,2]>180]      #CGM>180
AvgM180=func(cgmManual,c1)
#print(AvgM180)

c1=cgmManual.loc[cgmManual.iloc[:,2]>250]      #CGM>250
AvgM250=func(cgmManual,c1)
#print(AvgM250)

c1=cgmManual.loc[(cgmManual.iloc[:,2]>=70) & (cgmManual.iloc[:,2]<=180)]      #CGM>=70 & <=180
AvgM70to180=func(cgmManual,c1)
#print(AvgM70to180)

c1=cgmManual.loc[(cgmManual.iloc[:,2]>=70) & (cgmManual.iloc[:,2]<=150)]      #CGM>=70 & <=150
AvgM70to150=func(cgmManual,c1)
#print(AvgM70to150)

c1=cgmManual.loc[cgmManual.iloc[:,2]<70]      #CGM<70
AvgM70=func(cgmManual,c1)
#print(AvgM70)

c1=cgmManual.loc[cgmManual.iloc[:,2]<54]      #CGM<54
AvgM54=func(cgmManual,c1)
#print(AvgM54)

#AUTO
#cgmAuto.to_csv('Auto.csv')
c1=cgmAuto.loc[cgmAuto.iloc[:,2]>180]      #CGM>180
AvgA180=func(cgmAuto,c1)
#print(AvgA180)

c1=cgmAuto.loc[cgmAuto.iloc[:,2]>250]      #CGM>250
AvgA250=func(cgmAuto,c1)
#print(AvgA250)

c1=cgmAuto.loc[(cgmAuto.iloc[:,2]>=70) & (cgmAuto.iloc[:,2]<=180)]      #CGM>=70 & <=180
AvgA70to180=func(cgmAuto,c1)
#print(AvgA70to180)

c1=cgmAuto.loc[(cgmAuto.iloc[:,2]>=70) & (cgmAuto.iloc[:,2]<=150)]      #CGM>=70 & <=150
AvgA70to150=func(cgmAuto,c1)
#print(AvgA70to150)

c1=cgmAuto.loc[cgmAuto.iloc[:,2]<70]      #CGM<70
AvgA70=func(cgmAuto,c1)
#print(AvgA70)

c1=cgmAuto.loc[cgmAuto.iloc[:,2]<54]      #CGM<70
AvgA54=func(cgmAuto,c1)
#print(AvgA54)

#--------------------------------------------------------------------------------------------
#OVERNIGHT
#MANUAL
start=dt.strptime('00:00:00','%H:%M:%S').time()
end=dt.strptime('06:00:00','%H:%M:%S').time()

cgmManualNight=cgmManual.loc[(cgmManual['Time']>=start) & (cgmManual['Time']<end)]
#cgmManualNight.to_csv('ManualNight.csv')

c1=cgmManualNight.loc[cgmManualNight.iloc[:,2]>180]      #CGM>180
AvgMN180=func(cgmManualNight,c1)
#print(AvgAN180)

c1=cgmManualNight.loc[cgmManualNight.iloc[:,2]>250]      #CGM>250
AvgMN250=func(cgmManualNight,c1)
#print(AvgMN250)

c1=cgmManualNight.loc[(cgmManualNight.iloc[:,2]>=70) & (cgmManualNight.iloc[:,2]<=180)]      #CGM>=70 & <=180
AvgMN70to180=func(cgmManualNight,c1)
#print(AvgMN70to180)

c1=cgmManualNight.loc[(cgmManualNight.iloc[:,2]>=70) & (cgmManualNight.iloc[:,2]<=150)]      #CGM>=70 & <=150
AvgMN70to150=func(cgmManualNight,c1)
#print(AvgMN70to150)

c1=cgmManualNight.loc[cgmManualNight.iloc[:,2]<70]      #CGM<70
AvgMN70=func(cgmManualNight,c1)
#print(AvgMN70)

c1=cgmManualNight.loc[cgmManualNight.iloc[:,2]<54]      #CGM<54
AvgMN54=func(cgmManualNight,c1)
#print(AvgMN54)

#AUTO

cgmAutoNight=cgmAuto.loc[(cgmAuto['Time']>=start) & (cgmAuto['Time']<end)]
#cgmAutoNight.to_csv('AutoNight.csv')

c1=cgmAutoNight.loc[cgmAutoNight.iloc[:,2]>180]      #CGM>180
AvgAN180=func(cgmAutoNight,c1)
#print(AvgAN180)

c1=cgmAutoNight.loc[cgmAutoNight.iloc[:,2]>250]      #CGM>250
AvgAN250=func(cgmAutoNight,c1)
#print(AvgAN250)

c1=cgmAutoNight.loc[(cgmAutoNight.iloc[:,2]>=70) & (cgmAutoNight.iloc[:,2]<=180)]      #CGM>=70 & <=180
AvgAN70to180=func(cgmAutoNight,c1)
#print(AvgA70to180)

c1=cgmAutoNight.loc[(cgmAutoNight.iloc[:,2]>=70) & (cgmAutoNight.iloc[:,2]<=150)]      #CGM>=70 & <=150
AvgAN70to150=func(cgmAutoNight,c1)
#print(AvgA70to150)

c1=cgmAutoNight.loc[cgmAutoNight.iloc[:,2]<70]      #CGM<70
AvgAN70=func(cgmAutoNight,c1)
#print(AvgA70)

c1=cgmAutoNight.loc[cgmAutoNight.iloc[:,2]<54]      #CGM<70
AvgAN54=func(cgmAutoNight,c1)
#print(AvgA54)

#--------------------------------------------------------------------------------------------
#DAYTIME
start=dt.strptime('06:00:00','%H:%M:%S').time()
end=dt.strptime('23:59:00','%H:%M:%S').time()

cgmManualDay=cgmManual.loc[(cgmManual['Time']>=start) & (cgmManual['Time']<=end)]
#cgmManualDay.to_csv('ManualDay.csv')

c1=cgmManualDay.loc[cgmManualDay.iloc[:,2]>180]      #CGM>180
AvgMD180=func(cgmManualDay,c1)
#print(AvgMN180)

c1=cgmManualDay.loc[cgmManualDay.iloc[:,2]>250]      #CGM>250
AvgMD250=func(cgmManualDay,c1)
#print(AvgMN250)

c1=cgmManualDay.loc[(cgmManualDay.iloc[:,2]>=70) & (cgmManualDay.iloc[:,2]<=180)]      #CGM>=70 & <=180
AvgMD70to180=func(cgmManualDay,c1)
#print(AvgMN70to180)

c1=cgmManualDay.loc[(cgmManualDay.iloc[:,2]>=70) & (cgmManualDay.iloc[:,2]<=150)]      #CGM>=70 & <=150
AvgMD70to150=func(cgmManualDay,c1)
#print(AvgMN70to150)

c1=cgmManualDay.loc[cgmManualDay.iloc[:,2]<70]      #CGM<70
AvgMD70=func(cgmManualDay,c1)
#print(AvgMN70)

c1=cgmManualDay.loc[cgmManualDay.iloc[:,2]<54]      #CGM<54
AvgMD54=func(cgmManualDay,c1)
#print(AvgMN54)

#AUTO
cgmAutoDay=cgmAuto.loc[(cgmAuto['Time']>=start) & (cgmAuto['Time']<end)]
#cgmAutoDay.to_csv('AutoDay.csv')

c1=cgmAutoDay.loc[cgmAutoDay.iloc[:,2]>180]      #CGM>180
AvgAD180=func(cgmAutoDay,c1)
#print(AvgA180)

c1=cgmAutoDay.loc[cgmAutoDay.iloc[:,2]>250]      #CGM>250
AvgAD250=func(cgmAutoDay,c1)
#print(AvgA250)

c1=cgmAutoDay.loc[(cgmAutoDay.iloc[:,2]>=70) & (cgmAutoDay.iloc[:,2]<=180)]      #CGM>=70 & <=180
AvgAD70to180=func(cgmAutoDay,c1)
#print(AvgA70to180)

c1=cgmAutoDay.loc[(cgmAutoDay.iloc[:,2]>=70) & (cgmAutoDay.iloc[:,2]<=150)]      #CGM>=70 & <=150
AvgAD70to150=func(cgmAutoDay,c1)
#print(AvgA70to150)

c1=cgmAutoDay.loc[cgmAutoDay.iloc[:,2]<70]      #CGM<70
AvgAD70=func(cgmAutoDay,c1)
#print(AvgAD70)

c1=cgmAutoDay.loc[cgmAutoDay.iloc[:,2]<54]      #CGM<54
AvgAD54=func(cgmAutoDay,c1)
#print(AvgAD54)

res1={'':['Manual mode','Auto mode'],
     'OverNight (CGM > 180 mg/dL)':[AvgMN180,AvgAN180],
     'OverNight (CGM > 250 mg/dL)':[AvgMN250,AvgAN250],
     'OverNight (CGM >= 70 mg/dL and CGM <= 180 mg/dL)':[AvgMN70to180,AvgAN70to180],
     'OverNight (CGM >= 70 mg/dL and CGM <= 150 mg/dL)':[AvgMN70to150,AvgAN70to150],
     'OverNight (CGM < 70 mg/dL)':[AvgMN70,AvgAN70],
     'OverNight (CGM < 54 mg/dL)':[AvgMN54,AvgAN54],
     'Day (CGM > 180 mg/dL)':[AvgMD180,AvgAD180],
     'Day (CGM > 250 mg/dL)':[AvgMD250,AvgAD250],
     'Day (CGM >= 70 mg/dL and CGM <= 180 mg/dL)':[AvgMD70to180,AvgAD70to180],
     'Day (CGM >= 70 mg/dL and CGM <= 150 mg/dL)':[AvgMD70to150,AvgAD70to150],
     'Day (CGM < 70 mg/dL)':[AvgMD70,AvgAD70],
     'Day (CGM < 54 mg/dL)':[AvgMD54,AvgAD54],
     'Whole Day (CGM > 180 mg/dL)':[AvgM180,AvgA180],
     'Whole Day (CGM > 250 mg/dL)':[AvgM250,AvgA250],
     'Whole Day (CGM >= 70 mg/dL and CGM <= 180 mg/dL)':[AvgM70to180,AvgA70to180],
     'Whole Day (CGM >= 70 mg/dL and CGM <= 150 mg/dL)':[AvgM70to150,AvgA70to150],
     'Whole Day (CGM < 70 mg/dL)':[AvgM70,AvgA70],
     'Whole Day (CGM < 54 mg/dL)':[AvgM54,AvgA54]
     }
finalres=pd.DataFrame(res1)
finalres.to_csv('Results.csv')
print('Done! Check the Results.csv file.')

import csv
import datetime
import pandas as pd
import numpy as np
import math
import os
from operator import itemgetter

INSULIN_PATH = 'InsulinData.csv'
CGM_PATH = 'CGMData.csv'
NO_OF_RECORDS_PER_DAY = 288


def load_csv_file(csv_path, select_column):
    csv_list = []
    with open(csv_path, newline='') as csvfile:
        data = csv.DictReader(csvfile)
        rowid = 0
        for row in data:
            rowid += 1
            date_time = datetime.datetime.strptime(row['Date'] + ' ' + row['Time'], '%m/%d/%Y %H:%M:%S')
            csv_list.append([rowid, row['Index'], datetime.datetime.timestamp(date_time),
                             row['Date'], row['Time'], row[select_column]])
    csv_list = sorted(csv_list, key=itemgetter(2))
    return csv_list


def analyse_data():
    insulin_list = load_csv_file(INSULIN_PATH, 'BWZ Carb Input (grams)')
    cgm_list = load_csv_file(CGM_PATH, 'Sensor Glucose (mg/dL)')
    meal_start = ''
    Tm = np.NaN
    meal_end = np.NaN
    no_meal_start = np.NaN
    Tend = np.NaN
    no_meal_end = np.NaN
    meal_time_list = []
    no_meal_time_list = []

    for insulin_data in insulin_list:
        if insulin_data[5] != '':
            if math.isnan(Tm) and math.isnan(Tend):
                Tm = insulin_data[2]
                Tend = datetime.datetime.timestamp(datetime.datetime.fromtimestamp(Tm) + datetime.timedelta(hours=2))
                no_meal_start = np.NaN
                no_meal_end = np.NaN
            elif Tm != np.NaN and Tend != np.NaN:
                if insulin_data[2] >= Tend:
                    meal_time_list.append(Tm)
                    Tm = np.NaN
                    Tend = np.NaN
                else:
                    Tm = insulin_data[2]
                    Tend = datetime.datetime.timestamp(
                        datetime.datetime.fromtimestamp(Tm) + datetime.timedelta(hours=2))
        elif not math.isnan(Tm) and not math.isnan(Tend):
            if insulin_data[2] >= Tend:
                meal_time_list.append(Tm)
                Tm = np.NaN
                Tend = np.NaN
        elif not math.isnan(no_meal_start) and not math.isnan(no_meal_end) and math.isnan(Tm) and math.isnan(Tend):
            if insulin_data[2] >= no_meal_end:
                no_meal_time_list.append(no_meal_start)
                no_meal_start = np.NaN
                no_meal_end = np.NaN
        elif math.isnan(no_meal_start) and math.isnan(no_meal_end) and math.isnan(Tm) and math.isnan(Tend):
            no_meal_start = insulin_data[2]
            no_meal_end = datetime.datetime.timestamp(
                datetime.datetime.fromtimestamp(no_meal_start) + datetime.timedelta(hours=2))

    meal_map_list = []
    no_meal_map_list = []
    # for meal data
    for meal_time in meal_time_list:
        cgm_bunch = []
        meal_point_start = datetime.datetime.timestamp(
            datetime.datetime.fromtimestamp(meal_time) - datetime.timedelta(hours=0.5))
        meal_point_end = datetime.datetime.timestamp(
            datetime.datetime.fromtimestamp(meal_time) + datetime.timedelta(hours=2))
        for cgm_data in cgm_list:
            if meal_point_start < cgm_data[2] <= meal_point_end and cgm_data[5] != '':
                cgm_bunch.append(cgm_data[5])
            if cgm_data[2] > meal_point_end:
                break
        if len(cgm_bunch) >= 25:
            meal_map_list.append(cgm_bunch[0:25])

    # for no meal data
    for no_meal_time in no_meal_time_list:
        cgm_bunch = []
        no_meal_point_start = no_meal_time
        no_meal_point_end = datetime.datetime.timestamp(
            datetime.datetime.fromtimestamp(no_meal_time) + datetime.timedelta(hours=2))
        for cgm_data in cgm_list:
            if cgm_data[2] > no_meal_point_start and cgm_data[5] != '':
                cgm_bunch.append(cgm_data[5])
            if cgm_data[2] >= no_meal_point_end:
                break
        if len(cgm_bunch) >= 25:
            no_meal_map_list.append(cgm_bunch[0:25])

   # chunked_data = [meal_map_list[i:i + 51] for i in range(0, len(meal_map_list), 51)]
   # for k in range(0, len(chunked_data)):
        csv_file = f"MealSet.csv"
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(meal_map_list)
        except IOError:
            print("I/O error")

   # chunked_data = [no_meal_map_list[i:i + 51] for i in range(0, len(no_meal_map_list), 51)]
   # for k in range(0, len(chunked_data)):
        csv_file = f"NoMealSet.csv"
        try:
            with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerows(no_meal_map_list)
        except IOError:
            print("I/O error")


if __name__ == '__main__':
    if not os.path.isfile(INSULIN_PATH):
        print(f'{INSULIN_PATH} missing. Please place it in the current folder.')
    elif not os.path.isfile(CGM_PATH):
        print(f'{CGM_PATH} missing. Please place it in the current folder.')
    else:
        print('Data analysis started......')
        analyse_data()
        print('Result file generated. Please check the current folder.')

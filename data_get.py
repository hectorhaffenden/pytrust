import pandas as pd
import numpy as np
import lxml.html as html
import math
import csv
import time
import requests


import requests
from bs4 import BeautifulSoup
import time
import json

import seaborn as sns
import matplotlib.pyplot as plt

from datetime import datetime

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 6.1; Win64; x64; rv:47.0) Gecko/20100101 Firefox/47.0"}




def get_url(company, country):
    if country == 'us':
        url = f'https://www.trustpilot.com/review/{company}'
    else:
        url = f'https://uk.trustpilot.com/review/{company}'
    return url

# Get number of pages
def get_number_of_pages(company, url):
    '''
    Input:
        company
        country = uk or us
    Output:
        Number of pages of reviews
    '''
    r=requests.get(url)
    soup = BeautifulSoup(r.text, 'lxml')

    review_count_h2=soup.find('h2',class_="header--inline").text

    review_count=int(review_count_h2.strip().split(' ')[0].strip().replace(',', ''))
    pages=int(math.ceil(review_count/20))
    print(f'Number of reviews: {review_count}, with {pages} pages')
    return pages

def get_data(url, number_of_pages, sleep_time = False):
    '''
    Input: 
        url:
        number_of_pages: from get_number_of_pages function
        sleep_time: if we want to sleep inbetween each call
    Output: pandas df
    '''
    final_list=[]
    for pg in range(1, number_of_pages):
        print(pg)
        if sleep_time:
            time.sleep(sleep_time)
        pg = url + '?page=' + str(pg)
        r=requests.get(pg)
        soup = BeautifulSoup(r.text, 'lxml')
        for paragraph in soup.find_all('section',class_='review__content'):
            try:
                title=paragraph.find('h2',class_='review-content__title').text.strip()
                content=paragraph.find('p',class_='review-content__text').text.strip()
                datedata= json.loads(paragraph.find('div', class_='review-content-header__dates').findAll(text=True)[1].replace('\n', ''))
                date=datedata['publishedDate'].split('T')[0]
                rating = paragraph.find('div', class_='star-rating star-rating--medium')                
                try:
                    rating = str(rating).split('stars: ')[1].split('" ')[0]
                except IndexError:
                    rating = str(rating).split('star: ')[1].split('" ')[0]
                try:
                    name = str(paragraph).split('consumerName":"')[1].split('",')[0]
                except IndexError:
                    name = 'UNVERIFIED'
                final_list.append([name, title, content, date, rating])
            except AttributeError:
                pass
    df = pd.DataFrame(final_list,columns=['name', 'title','content','date','rating'])
    # Expand dates
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d').astype('datetime64[ns]')
    df['year'] = df['date'].dt.year.astype('int16')
    df['month'] = df['date'].dt.month.astype('int16')
    df['week'] = df['date'].dt.week.astype('int16')
    df['day'] = df['date'].dt.day.astype('int16')
    df['wday'] = df['date'].dt.weekday.astype('int16')
    # Add rating number
    dict_ratings = {'Excellent': 5, 'Great': 4, 'Average': 3, 'Poor': 2, 'Bad': 1}
    df['num_stars'] = df['rating'].replace(dict_ratings)
    # Reorder columns nicely
    order_cols = ['date', 'year','month',
              'week', 'day', 'num_stars',
              'rating', 'name', 'title',
              'content']
    df.loc[:, order_cols]
    return df

def plot_reviews(df, company, window = 30):
    '''
    Plots a rolling mean of the ratings
    '''
    grp_by_day = df.groupby(['date']).mean().reset_index()
    grp_by_day['rolling_mean_week'] = grp_by_day['num_stars'].rolling(window).mean()
    # Plot the results
    a4_dims = (15, 4)
    fig, ax = plt.subplots(figsize=a4_dims)
    ax.set_title(f'{company} - {window} window rolling mean of {df.shape[0]} customer reviews')
    sns.set(font_scale = 1.5)
    sns.lineplot(data = grp_by_day, x = 'date', y = 'rolling_mean_week', ax=ax) # Higher is better
    ax.set(xlabel='Time (daily)', ylabel='Avg customer reviews')
    plt.ylim([1, 5])
    plt.show()
    

    
    
def test_companies(companies, country):
    '''
    Input:
        company
        country
    Output:
        if no error, then all good
    '''
    for comp in companies:
        url = get_url(comp, country)
        r=requests.get(url)
        soup = BeautifulSoup(r.text, 'lxml')
        review_count_h2=soup.find('h2',class_="header--inline").text
        review_count=int(review_count_h2.strip().split(' ')[0].strip().replace(',', ''))
        print(f'{comp} works and has {review_count} reviews')
        
def run_and_save_data(company, country, sleep_time = False):
    '''
    Input: 
        company
        country
    Output:
        saves a csv file with data
    '''
    url = get_url(company, country)
    page_num = get_number_of_pages(company, url)
    data = get_data(url, number_of_pages = page_num, sleep_time = sleep_time)
    today = datetime.today().strftime('%Y%m%d')
    # clean full stops from name
    clean_comp_name = '_'.join(company.split('.'))
    path_final = f'data/{today}_{clean_comp_name}_trustpilot.csv'
    data.to_csv(path_final, index=False)
    print(path_final)
    
  
    
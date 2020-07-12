import pandas as pd
import numpy as np
import random
import datetime as dt

from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import seaborn as sns

from plotly import graph_objs as go
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from PIL import Image

from nltk.probability import FreqDist

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF



def plot_dist3(df, feature, title):
    # Creating a customized chart. and giving in figsize and everything.
    fig = plt.figure(constrained_layout=True, figsize=(18, 8))
    # Creating a grid of 3 cols and 3 rows.
    grid = gridspec.GridSpec(ncols=3, nrows=3, figure=fig)

    # Customizing the histogram grid.
    ax1 = fig.add_subplot(grid[0, :2])
    # Set the title.
    ax1.set_title('Histogram')
    # plot the histogram.
    sns.distplot(df.loc[:, feature],
                 hist=True,
                 kde=True,
                 ax=ax1,
                 color='#e74c3c')
    ax1.set(ylabel='Frequency')
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=20))

    # Customizing the ecdf_plot.
    ax2 = fig.add_subplot(grid[1, :2])
    # Set the title.
    ax2.set_title('Empirical CDF')
    # Plotting the ecdf_Plot.
    sns.distplot(df.loc[:, feature],
                 ax=ax2,
                 kde_kws={'cumulative': True},
                 hist_kws={'cumulative': True},
                 color='#e74c3c')
    ax2.xaxis.set_major_locator(MaxNLocator(nbins=20))
    ax2.set(ylabel='Cumulative Probability')

    # Customizing the Box Plot.
    ax3 = fig.add_subplot(grid[:, 2])
    # Set title.
    ax3.set_title('Box Plot')
    # Plotting the box plot.
    sns.boxplot(x=feature, data=df, orient='v', ax=ax3, color='#e74c3c')
    ax3.yaxis.set_major_locator(MaxNLocator(nbins=25))

    
    
    
    lims = 0, df.loc[:, feature].max()

    ax1.set_xlim(lims)
    ax2.set_xlim(lims)
    
    
    plt.suptitle(f'{title}', fontsize=18)
    
def plot_word_len_histogram(textno, textye):
    
    """A function for comparing average word length"""
    
    fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(18, 6), sharey=True)
    
    left = textno.str.split().apply(lambda x: [len(i) for i in x]).map(
        lambda x: np.mean(x))
    sns.distplot(left, ax=axes[0], color='#e74c3c')
    
    right = textye.str.split().apply(lambda x: [len(i) for i in x]).map(
        lambda x: np.mean(x))
    sns.distplot(right, ax=axes[1], color='#e74c3c')
    
    mx_l = max(left)
    mx_r = max(right)
    
    mi_l = min(left)
    mi_r = min(right)
    lims = min(mi_r, mi_l), max(mx_r, mx_l)
    axes[0].set_xlim(lims)
    axes[1].set_xlim(lims)
    axes[0].set_xlabel('Word Length')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Positive reviews')
    axes[1].set_xlabel('Word Length')
    axes[1].set_title('Negative reviews')
    
    fig.suptitle('Mean Word Lengths', fontsize=24, va='baseline')
    fig.tight_layout()
    
    
def plot_pie_chart(df):

    temp = df.groupby('num_stars')['title'].count().reset_index().sort_values('num_stars')
    labels = temp['num_stars'].values
    cols = ['lightblue', 'red', 'green', 'purple', 'orange']
    plt.pie(temp['title'], radius=2, autopct = '%0.1f%%',
            shadow = True, explode = [0.2,0,0,0,0.2],
            startangle = 0, labels = labels, colors = cols)
    plt.title('Pie chart of number of reviews with respective 1-5 star ratings', fontsize=18, y = 1.3)
    plt.show()
    
def plot_star_funnel(df):
    temp = df.groupby('num_stars')['title'].count().reset_index().sort_values('num_stars')
    fig = go.Figure(go.Funnelarea(
        text = temp['num_stars'],
        values = temp['title'],
        title = {'position': 'top center', 'text': 'Funnel chart of ratings'}
    ))
    
    fig.update_layout(
        titlefont=dict(
            family="InterFace",
            size=30,
        )
    )
    fig.show()
    
    
    
from sklearn.feature_extraction.text import CountVectorizer
def ngrams(df, n, title, mx_df = 0.9, content_or_title = 'content'):
    """
    A Function to plot most common ngrams
    content_or_title - use the content of the review, or the title
    mx_df: Ignore document frequency strictly higher than the given threshold
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    axes = axes.flatten()
    #plt.rcParams.update({'font.size': 25})
    for rate, j in zip([5, 1], axes):
        new = df[df['num_stars'] == rate][content_or_title].str.split()
        new = new.values.tolist()
        corpus = [word for i in new for word in i]

        def _get_top_ngram(corpus, n=None):
            #getting top ngrams
            vec = CountVectorizer(ngram_range=(n, n),
                                  max_df=mx_df,
                                  stop_words='english').fit(corpus)
            bag_of_words = vec.transform(corpus)
            sum_words = bag_of_words.sum(axis=0)
            words_freq = [(word, sum_words[0, idx])
                          for word, idx in vec.vocabulary_.items()]
            words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
            return words_freq[:15]

        top_n_bigrams = _get_top_ngram(df[df['num_stars'] == rate][content_or_title], n)[:15]
        x, y = map(list, zip(*top_n_bigrams))
        sns.barplot(x=y, y=x, palette='plasma', ax=j)

        
        
    title_font = {'fontname':'Arial', 'size':'24', 'color':'black', 'weight':'normal',
              'verticalalignment':'bottom'}
    lab_font = {'fontname':'Arial', 'size':'20'}
    
    
    axes[0].set_title(f'Positive reviews - for {content_or_title} of review', **title_font)
    axes[1].set_title(f'Negative reviews - for {content_or_title} of review', **title_font)
    axes[0].set_xlabel('Count', **lab_font)
    axes[0].set_ylabel('Words', **lab_font)
    axes[1].set_xlabel('Count', **lab_font)
    axes[1].set_ylabel('Words', **lab_font)
    
    axes[0].tick_params(axis='both', which='major', labelsize=15)
    axes[1].tick_params(axis='both', which='major', labelsize=15)
    
    #fig.suptitle(title, fontsize=24, va='baseline')
    plt.tight_layout()



def plot_wordcloud(df, max_words=400, max_font_size=120, figure_size=(24.0,16.0), 
                   title = None, title_size=40, image_color=False, image = 'basket.png',
                  more_stopwords = {'one', 'br', 'Po', 'th', 'sayi', 'fo', 'Unknown'}):
    comments_text = str(df.values)
    mask = np.array(Image.open('images/' + image))
    
    stopwords = set(STOPWORDS)
    stopwords = stopwords.union(more_stopwords)

    wordcloud = WordCloud(background_color='white',
                    stopwords = stopwords,
                    max_words = max_words,
                    max_font_size = max_font_size, 
                    random_state = 42,
                    mask = mask)
    wordcloud.generate(comments_text)
    
    plt.figure(figsize=figure_size)
    if image_color:
        image_colors = ImageColorGenerator(mask);
        plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation="bilinear");
        plt.title(title, fontdict={'size': title_size,  
                                  'verticalalignment': 'bottom'})
    else:
        plt.imshow(wordcloud);
        plt.title(title, fontdict={'size': title_size, 'color': 'green', 
                                  'verticalalignment': 'bottom'})
    plt.axis('off');
    plt.tight_layout()
    
    

    
    
    


def time_series_slider(df, window = 7, add_count = False, add_var = False, add_kurt = False):
    mean_grp = df.groupby('date')['num_stars'].mean().rolling(window).mean().reset_index().iloc[window:,:]
    # Create figure
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(x=list(mean_grp['date']), y=list(mean_grp['num_stars']), name="rolling mean"))
    
    if add_count:
        mean_grp = df.groupby('date')['num_stars'].count().rolling(window).mean().reset_index().iloc[window:,:]
        fig.add_trace(go.Scatter(x=list(mean_grp['date']), y=list(mean_grp['num_stars']), name="rolling count"),
                     secondary_y = True)
    
    if add_var:
        mean_grp = df.groupby('date')['num_stars'].mean().rolling(window).std().reset_index().iloc[window:,:]
        fig.add_trace(go.Scatter(x=list(mean_grp['date']), y=list(mean_grp['num_stars']), name="rolling std"),
                     secondary_y = True)
    
    if add_kurt:
        mean_grp = df.groupby('date')['num_stars'].mean().rolling(window).kurt().reset_index().iloc[window:,:]
        fig.add_trace(go.Scatter(x=list(mean_grp['date']), y=list(mean_grp['num_stars']), name="rolling kurtosis"),
                     secondary_y = True)
    
    # Set title
    fig.update_layout(
        title_text="Time series with range slider and selectors"
    )

    # Add range slider
    fig.update_layout(title="Rolling mean of average reviews",
                      xaxis_title="Date",
                      yaxis_title="Value",
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1,
                         label="1m",
                         step="month",
                         stepmode="backward"),
                    dict(count=6,
                         label="6m",
                         step="month",
                         stepmode="backward"),
                    dict(count=1,
                         label="YTD",
                         step="year",
                         stepmode="todate"),
                    dict(count=1,
                         label="1y",
                         step="year",
                         stepmode="backward"),
                    dict(step="all")
                ])
            ),
            rangeslider=dict(
                visible=True
            ),
            type="date"
        ),
        yaxis=dict(
           autorange = True,
           fixedrange= False
        )
    )
    fig['layout']['yaxis'].update(title = 'Number of stars', range = [0, 5], autorange = False)
    fig.show()
    
    

def display_topics(text, no_top_words, topic, components = 10):
    """
    A function for determining the topics present in our corpus with nmf
    """
    no_top_words = no_top_words
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.90, min_df=25, max_features=5000, use_idf=True)
    tfidf = tfidf_vectorizer.fit_transform(text)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    doc_term_matrix_tfidf = pd.DataFrame(
        tfidf.toarray(), columns=list(tfidf_feature_names))
    nmf = NMF(n_components=components, random_state=0,
              alpha=.1, init='nndsvd', max_iter = 5000).fit(tfidf)
    print(topic)
    for topic_idx, topic in enumerate(nmf.components_):
        print('Topic %d:' % (topic_idx+1))
        print(' '.join([tfidf_feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
    
    
    
def trend_barplots(df, mean_or_count = 'mean', plots = ['wday', 'day', 'week', 'month', 'year']):
    '''
    Input: 
        df: data
        mean_or_count: Takes values mean, or count, and decides what to plot
        plots: by which grouping do you want the plots
    Output: 
        displays barplots
    '''
    # Prep data
    for col in plots:
        if col == 'wday':
            df[col] = pd.to_datetime(df['date'], format='%Y-%m-%d').astype('datetime64[ns]').dt.weekday
        
        if mean_or_count == 'mean':
            grp_wday = df.groupby([col])['num_stars'].mean()
            lims = (1,5)
            title_besp = f'Average stars by {col}'
        elif mean_or_count == 'count':
            grp_wday = df.groupby([col])['num_stars'].count()
            # Set limit to 5% above the max
            lims = (0,grp_wday.max() + round(grp_wday.max() / 20))
            title_besp = f'Count of reviews by {col}'
        plt.figure(figsize=(10,6))
        sns.barplot(x=grp_wday.index.values,
                    y=grp_wday.values, palette='plasma')
        plt.ylim(lims)
        plt.title(title_besp, fontsize=20)
        plt.xlabel(f'{col}', fontsize=18)
        plt.ylabel('Number of stars', fontsize=16)
        plt.tight_layout()
        plt.show()
    
    
    
    
    
    
    
    
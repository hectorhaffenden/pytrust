# pytrust
<b>The python package to scrape, transform and analyse data from Trustpilot.</b>

- The aim is to run NLP analysis that is generalisable to many tasks, especially reviews.


<b>This package is still in development</b>


Package has two main components, scraping scripts, and analysis scripts
1. Scraping - This comprises the collection and cleaning of data
2. Analysis - This includes all analysis, from removing stop words, NLP modelling, visualisations etc...



### Notes
- You are able to collect data from multiple companies, then evaluate them together, or separately
	- There are detailed examples of the above two cases in the form of a jupyter notebook



#### Disclaimer
- I don't have any connection with Trustpilot and this project is neither approved or endorsed by them.
- Data from Trustpilot is publicly accessible (without logging in to the website) at the moment it is collected.
- This package was created for educational purposes.


#### Resources
I've learnt a lot from the following resources (mainly around NLP analysis)
- https://www.kaggle.com/roshansharma/amazon-alexa-reviews/data
- https://www.kaggle.com/nicapotato/womens-ecommerce-clothing-reviews



### Todo
<b>General</b>
- Move code to modules
- Switch from functions to classes
- Peer review structure
- Create folders and text files required, if they don't exist, on local machines?

<b>Scraping</b>
- Add ability to store data in SQLite

<b>Analysis</b>
- Add more powerful models to classification
- 3D plot of stars vs review length vs company
- Work out a better way to switch between looking at one vs multiple companies
- Add options for working with companies with different skewed datasets (e.g. 95% 5 stars vs 50/50)
- Find a consistent theme

import streamlit as st
import pandas as pd
from utils import *
from parameters import *
from graphs import *
from database import Database_Manager, DBAmbienceKeywords, DBDrinks, DBMenu, DBProductKeywords, DBServiceKeywords
from ai_classifier import ArtificialWalla
from translator_walla import Translator
from streamlit_pills import pills
# DB connection
def fetch_data_from_db(name = 'pages/details.db'):
   db = Database_Manager(name)
   data = db.view()
   if len(data) == 0:
      return None
   return data

def process_direct_feedback(direct_feedback: list, df: pd.DataFrame):
   '''
   This function is used to process the direct feedback and add it to the dataframe

   ---

   params:
      direct_feedback : list of direct feedback files (it should contain only one file.xlsx)
      df: the dataframe that contains the data from the database to which we will add the direct feedback
   
   return:
      df_direct_feedback: the final dataframe with the direct feedback added
   '''
   st.write('Direct Feedback: There are emails as well')
   df_direct_feedback = pd.read_excel(direct_feedback[0])
   # change column names: CAFE == 'Reservation: Venue', 'DATE RECEIVED' == 'Date Submitted', 'FEEDBACK' == 'Feedback: Feedback', 'Source' == 'Platform'
   df_direct_feedback = df_direct_feedback.rename(columns={'CAFE': 'Reservation: Venue', 'DATE RECEIVED': 'Date Submitted', 'FEEDBACK': 'Details', 'Source': 'Platform'})
   # keep only row with "Details" not empty
   df_direct_feedback = df_direct_feedback[df_direct_feedback['Details'] != '' ]
   # add all the columns that are inside the other df
   columns_to_add = df.columns.tolist()
   for col in df_direct_feedback.columns.tolist():
      if col not in columns_to_add:
         columns_to_add.append(col)

   # add the columns that are not in the df_direct_feedback
   for col in columns_to_add:
      if col not in df_direct_feedback.columns.tolist():
         df_direct_feedback[col] = ["" for i in range(len(df_direct_feedback))]

   df_direct_feedback = df_direct_feedback[df.columns]
   # keep only the not empty Details 
   df_direct_feedback = df_direct_feedback[df_direct_feedback['Details'].astype(str) != 'nan']
   # set the same type as the df columns
   df_direct_feedback["Label: Dishoom"] = ["" for i in range(len(df_direct_feedback))]
   # transform the date into datetime

   df_direct_feedback["Date Submitted"] = df_direct_feedback["Date Submitted"].apply(lambda x: str(pd.to_datetime(x).date()))
      # get week, month and day name from the date if there is one
   df_direct_feedback["Week"] = df_direct_feedback.apply(lambda_for_week, axis=1)
   df_direct_feedback["Month"] = df_direct_feedback.apply(lambda_for_month, axis=1)
   df_direct_feedback["Day_Name"] = df_direct_feedback.apply(lambda_for_day_name, axis=1)
   df_direct_feedback["Day_Part"] = df_direct_feedback.apply(lambda_for_day_part, axis=1)
   # convert the datetime into date
   # add year
   df_direct_feedback["Year"] = df_direct_feedback["Date Submitted"].apply(lambda x: str(pd.to_datetime(x).year))
   # add the source
   df_direct_feedback["Source"] = "Direct Feedback"
   # add week year
   df_direct_feedback["Week_Year"] = df_direct_feedback["Week"] + "W" + df_direct_feedback["Year"]
   # add month year
   df_direct_feedback["Month_Year"] = df_direct_feedback["Month"] + "M" + df_direct_feedback["Year"]
   # set date for filter
   df_direct_feedback["date_for_filter"] = df_direct_feedback["Date Submitted"]
   # set day part 
   df_direct_feedback["Day_Part"] = df_direct_feedback["Day_Part"].apply(lambda x: get_day_part(x))
   # set the thumbs up and thumbs down 👍 👎 columns as False
   df_direct_feedback["👍"] = [False for i in range(len(df_direct_feedback))]
   df_direct_feedback["👎"] = [False for i in range(len(df_direct_feedback))]
   df = pd.concat([df, df_direct_feedback], axis=0)
   # add the new columns for the scoring
   df['New Overall Rating'] = 1
   df['New Food Rating'] = 1
   df['New Drink Rating'] = 1
   df['New Service Rating'] = 1
   df['New Ambience Rating'] = 1
   return df
      
def preprocess_single_df(df):
   # 3. Prepare the dataframes: 
   # add Reservation: Venue when empty (name of the restaurant)
   venue = df["Reservation: Venue"].unique().tolist()
   venue = [v for v in venue if str(v) != 'nan'][0]
   venue = str(venue).replace("'", "")
   df["Reservation: Venue"] = venue
   # add all the columns that we are going to use
   df["Label: Dishoom"] = ["" for i in range(len(df))]
   df['👍'] = False 
   df['👎'] = False
   df['💡'] = False    
   df['Source'] = df['Platform']
   # ADD: Week, Month, Day_Name, Day_Part, Year, Week_Year, Month_Year, date_for_filter
   # there is this sign / and the opposite \ in the date, so we need to check for both
   df["Week"] = df.apply(lambda_for_week, axis=1)
   df["Month"] = df.apply(lambda_for_month, axis=1)
   df["Day_Name"] = df.apply(lambda_for_day_name, axis=1)
   df['Day_Part'] = df.apply(lambda_for_day_part, axis=1)
   df['Year'] = df.apply(lambda x: str(pd.to_datetime(x['Date Submitted']).year) if x['Reservation: Date'] in empty else str(pd.to_datetime(x['Reservation: Date']).year), axis=1)
   df['Week_Year'] = df.apply(lambda x: x['Week'] + 'W' + x['Year'], axis=1)
   df['Month_Year'] = df.apply(lambda x: x['Month'] + 'M' + x['Year'], axis=1)
   df['date_for_filter'] = df.apply(lambda x: str(pd.to_datetime(x['Date Submitted']).date()) if x['Reservation: Date'] in empty else str(pd.to_datetime(x['Reservation: Date']).date()), axis=1)
   df['Suggested to Friend'] = df['Feedback: Recommend to Friend'].apply(lambda x: x if x == 'Yes' or x == 'No' else 'Not Specified')
   # initialize the new scoring columns
   df['New Overall Rating'] = 1
   df['New Food Rating'] = 1
   df['New Drink Rating'] = 1
   df['New Service Rating'] = 1
   df['New Ambience Rating'] = 1
   # set all scores to 0
   return df
      
def create_data_from_uploaded_file():
   '''
   In this function we will create the dataframe from the uploaded file,
   preparing it for the AI model to predict the sentiment.

   '''
   # read multiple files
   files = st.file_uploader("Upload Excel", type="xlsx", accept_multiple_files=True, key='upload')
   
   if files is not None:
      # 1. When received multiple files, we need to check if there is a direct feedback file
      direct_feedback = [f for f in files if f.name == 'Direct_Feedback.xlsx']
      files = [f for f in files if f.name != 'Direct_Feedback.xlsx']
      
      # 2. Read all the files and store them in a list
      dfs = [pd.read_excel(f) for f in files]

      individual_step = 95//len(dfs)
      progress_text = 'Uploading Data'
      my_bar = st.progress(0, text=progress_text)

      for i, df in enumerate(dfs):
         my_bar.progress(int((i+1) * individual_step), text=progress_text)
         df = preprocess_single_df(df)

      my_bar.progress(95, text='Now Processing the data')
      df = pd.concat(dfs, ignore_index=True)

      # add the direct feedback file
      if len(direct_feedback) == 1:
         df = process_direct_feedback(direct_feedback, df)
         #st.write(df)

      # Dividing the data into two dfs:  one with empty details and one with not empty details
      df_not_empty = df[df['Details'].astype(str) != 'nan']
      df_empty = df[df['Details'].astype(str) == 'nan']

      # drop duplicates:
      # the problem is that the details are not the same but the stripped details are the same 
      # (stripped details are the details without spaces and new lines)
      #df_not_empty['Stripped_det'] = df_not_empty['Details'].apply(lambda x: x.replace(' ', '').replace('\n', '').replace('\r', '').strip())
      #df_not_empty = df_not_empty.drop_duplicates(subset=['Stripped_det'])
      #df_not_empty = df_not_empty.drop(columns=['Stripped_det'])

      # now we have to concat the two dfs
      df = pd.concat([df_not_empty, df_empty], ignore_index=True)
      my_bar.progress(100, text='')
      return df

def add_additional_data_from_uploaded_file(original_df: pd.DataFrame, files = None):
   if files is not None:
      # 1. When received multiple files, we need to check if there is a direct feedback file
      direct_feedback = [f for f in files if f.name == 'Direct_Feedback.xlsx']
      files = [f for f in files if f.name != 'Direct_Feedback.xlsx']
      
      # 2. Read all the files and store them in a list
      dfs = [pd.read_excel(f) for f in files]

      individual_step = 95//len(dfs)
      progress_text = 'Uploading Data'
      my_bar = st.progress(0, text=progress_text)

      for i, df in enumerate(dfs):
         my_bar.progress(int((i+1) * individual_step), text=progress_text)
         # 3. Prepare the dataframes: 
         # add Reservation: Venue when empty (name of the restaurant)
         venue = df["Reservation: Venue"].unique().tolist()
         venue = [v for v in venue if str(v) != 'nan'][0]
         venue = str(venue).replace("'", "")
         df["Reservation: Venue"] = venue
         # add all the columns that we are going to use
         df["Label: Dishoom"] = ["" for i in range(len(df))]
         df['👍'] = False 
         df['👎'] = False
         df['💡'] = False    
         df['Source'] = df['Platform']
         # ADD: Week, Month, Day_Name, Day_Part, Year, Week_Year, Month_Year, date_for_filter
         # there is this sign / and the opposite \ in the date, so we need to check for both
         df["Week"] = df.apply(lambda_for_week, axis=1)
         df["Month"] = df.apply(lambda_for_month, axis=1)
         df["Day_Name"] = df.apply(lambda_for_day_name, axis=1)
         df['Day_Part'] = df.apply(lambda_for_day_part, axis=1)
         df['Year'] = df.apply(lambda x: str(pd.to_datetime(x['Date Submitted']).year) if x['Reservation: Date'] in empty else str(pd.to_datetime(x['Reservation: Date']).year), axis=1)
         df['Week_Year'] = df.apply(lambda x: x['Week'] + 'W' + x['Year'], axis=1)
         df['Month_Year'] = df.apply(lambda x: x['Month'] + 'M' + x['Year'], axis=1)
         df['date_for_filter'] = df.apply(lambda x: str(pd.to_datetime(x['Date Submitted']).date()) if x['Reservation: Date'] in empty else str(pd.to_datetime(x['Reservation: Date']).date()), axis=1)
         df['Suggested to Friend'] = df['Feedback: Recommend to Friend'].apply(lambda x: x if x == 'Yes' or x == 'No' else 'Not Specified')
         
         # initialize the new scoring columns
         df['New Overall Rating'] = 1
         df['New Food Rating'] = 1
         df['New Drink Rating'] = 1
         df['New Service Rating'] = 1
         df['New Ambience Rating'] = 1
      
      my_bar.progress(95, text='Now Processing the data')

      # concat the dfs into one
      df = pd.concat(dfs, ignore_index=True)

      # add the direct feedback file
      if len(direct_feedback) == 1:
         df = process_direct_feedback(direct_feedback, df)
         #st.write(df)


      # Dividing the data into two dfs:  one with empty details and one with not empty details
      df_not_empty = df[df['Details'].astype(str) != 'nan']
      df_empty = df[df['Details'].astype(str) == 'nan']

      # drop duplicates:
      # the problem is that the details are not the same but the stripped details are the same 
      # (stripped details are the details without spaces and new lines)
      #df_not_empty['Stripped_det'] = df_not_empty['Details'].apply(lambda x: x.replace(' ', '').replace('\n', '').replace('\r', '').strip())
      #df_not_empty = df_not_empty.drop_duplicates(subset=['Stripped_det'])
      #df_not_empty = df_not_empty.drop(columns=['Stripped_det'])

      # now we have to concat the two dfs
      df = pd.concat([df_not_empty, df_empty], ignore_index=True)
      # add the last five to the bar
      my_bar.progress(100, text='')
      # merge with the original df
      original_df = pd.concat([original_df, df], ignore_index=True)
      st.write(original_df)
      # now save it to the database
      # process the data
      return original_df

# main class
class FeedBackHelper:
    '''
    This class will create the main application interface
    '''
    def __init__(self, db_name, name_user):
        self.name_user = name_user
        self.walla =  ArtificialWalla()
        self.translator = Translator()
        self.title = 'Feedback Reviewer'
        self.db_name = db_name
        self.db_main_manager = Database_Manager(self.db_name)
        self.get_data()

    def get_data(self):
        try:
          self.data = self.db_main_manager.get_main_db_from_venue()
          if self.data is not None:
            self.df = pd.DataFrame(self.data, columns=['idx'] + Database_Manager.COLUMNS_FOR_CREATION)
          else:
               df = create_data_from_uploaded_file()
               self.df = self.process_data(df)
               # save to database
               self.df = self.save_to_db()
               self.db_main_manager.create_database_for_each_venue()
        except:
            df = create_data_from_uploaded_file()
            self.df = self.process_data(df)
            # save to database
            self.df = self.save_to_db()
            self.db_main_manager.create_database_for_each_venue()
        return self.df
    
    def _preprocessing(self, data):
      '''
      Here we will do the cleaning of the data
      
      - Just filling na with empty string
      ---
      Parameters:
      
         data: pandas dataframe

      Returns:
         data: pandas dataframe
      ---
      '''
      data = data.fillna('')
      return data

    def _classifing(self, data):
      '''
      Here we will do the classification of the data
      - Sentiment
      - Confidence
      - Menu Item
      - Keywords
      - Drink Item
      '''
      for index, row in data.iterrows():
         sentiment, confidence, menu_items, keywords_, drinks_items = self.walla.classify_review(row['Details'])
         columns_for_rating = ['Overall Rating','Feedback: Food Rating', 'Feedback: Drink Rating','Feedback: Service Rating', 'Feedback: Ambience Rating']
         values = [row['Overall Rating'], row['Feedback: Food Rating'], row['Feedback: Drink Rating'], row['Feedback: Service Rating'], row['Feedback: Ambience Rating']]
         # replace 5.0 with 5
         #as strings
         values = [str(v) for v in values]
         # replace 5.0 with 5
         values = [v.replace('.0', '') for v in values]
         # if all 5 or 0, then the sentiment is positive
         not_positive_values = ['1', '2', '3', '4']
         if all(v not in not_positive_values for v in values):
            sentiment = 'POSITIVE'
            confidence = 1
         else:
            sentiment = 'NEGATIVE'
            confidence = 1         
         data.loc[index, 'Sentiment'] = sentiment
         data.loc[index, 'Confidence'] = confidence
         data.loc[index, 'Menu Item'] = ' '.join(menu_items)
         data.loc[index, 'Keywords'] = ' '.join(keywords_)
         data.loc[index, 'Drink Item'] = ' '.join(drinks_items)

      return data
   
    def process_data(self, df, db_name = None):
         '''
         Here we run the actual transformation of the data
         '''
         df = self._preprocessing(df)
         self.df = self._classifing(df)
         self.df = rescoring(self.df)
         return self.df
    
    def save_to_db(self, db_name = None):
         '''
         Here we save the data to the database
         '''
         if db_name == None:
            save_to_db(self.df, Database_Manager.COLUMNS_FOR_CREATION, self.db_name)
         else:
            save_to_db(self.df, Database_Manager.COLUMNS_FOR_CREATION, db_name)
         return self.df

    def plot(self):
      # fill na in reservation date with the date of the review

      final = self.to_plot
      container_keywords = st.sidebar.container()
      with st.expander('Graphs 📉', expanded=False): # graph emoji 📈 or 📊 or 📉 
         tabs = st.tabs(['Graphs', 'Keywords', 'Pie Chart', 'Source Analysis', 'Day Analysis', 'Hour Analysis', 'Week Analysis', 'Month Analysis', 'Totals'])

         with tabs[0]:
            create_timeseries_graph(final, self.main_c)

         with tabs[1]:
            create_graph_keywords_as_a_whole(final, container = container_keywords)

         with tabs[2]:
            create_pie_chart(final)
         
         with tabs[3]:
            create_graph_for_source_analysis(final)

         with tabs[4]:
            create_graph_for_day_analysis(final)
         
         with tabs[5]:
            create_graph_for_hour_analysis(final)

         with tabs[6]:
            create_graph_for_week_analysis(final)

         with tabs[7]:
            create_graph_for_month_analysis(final)
         with tabs[8]:
            create_chart_totals_labels(final, self.main_c)

      create_container_for_each_sentiment(df = final, df_empty=self.df_without_review)

    def filter_search_bar(self, search_bar):
      if search_bar != '':
         # If the search bar is not empty, filter the dataframe
         # if search bar contains more than one word, split the words at "," and search for the ones that contains both
         if ',' in search_bar:
            search_bar = search_bar.split(',')
            # iterate over the list of words
            for word in search_bar:
               # if the word is not empty, filter the dataframe
               if word != '':
                  self.df = self.df[self.df['Details'].str.contains(word, case=False)]
         else:
            self.df = self.df[self.df['Details'].str.contains(search_bar, case=False)]

         # make sure the dataframe is not empty
         if self.df.shape[0] == 0:
            st.error('No results found. Please try again.')
            st.stop()
      return self.df
    
    def filter_by_date(self, df, start_date, end_date):
      try:
         # 5. Filter the dataframe if the dates are not None
         if start_date != None and end_date != None:
            # 4. Input needs to be transformed to datetime
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            # If both dates are not None, filter the dataframe
            self.df['date_for_filter'] = pd.to_datetime(self.df['date_for_filter'])
            self.df = self.df[(self.df['date_for_filter'] >= start_date) & (self.df['date_for_filter'] <= end_date)]
            # re-tranform to string
            self.df['date_for_filter'] = self.df['date_for_filter'].dt.strftime('%Y-%m-%d')
      except:
         st.warning('Please select a date range')
         st.stop()
      return self.df, start_date, end_date
   
    def filter_by_café(self, restaurant_name, lookUpCafe):
      if restaurant_name != '*':
         # get the name of the restaurant
         restaurant_name = list(lookUpCafe.keys())[list(lookUpCafe.values()).index(restaurant_name)]
         name_choosen_db = 'pages/' + restaurant_name + '.db'
         self.df = pd.DataFrame(fetch_data_from_db(name=name_choosen_db), columns=['idx'] + Database_Manager.COLUMNS_FOR_CREATION)
      else:
         name_choosen_db = '*'
      return self.df, name_choosen_db
   
    def UploadNewData(self):
      '''
      1. Show an error
      2. We are going to get all the data current
      '''
      # Show an error if the dataframe is empty
      st.error('No results found. Please try again.')
      old_data = self.db_main_manager.get_main_db_from_venue()
      if old_data is not None:
         old_data = pd.DataFrame(old_data, columns=['idx'] + Database_Manager.COLUMNS_FOR_CREATION)
         #st.write(old_data['Reservation: Venue'].unique().tolist())
      files = st.file_uploader("Upload Excel", type="xlsx", accept_multiple_files=True, key = 'upload_2')
      if len(files) > 0:
         files = [f for f in files if f.name != 'Direct_Feedback.xlsx']
         data = add_additional_data_from_uploaded_file(old_data, files)
         self.df = self.process_data(data)
         self.save_to_db('pages/details.db')
         st.success('Data uploaded successfully to {}'.format(self.db_name))
         self.db_main_manager.create_database_for_each_venue()
         st.stop()
      else:
         st.stop()

    def run(self):
      '''
      Here we will run the app

      ---
      1. Set Logo of the page
      2. Search bar
      3. Date Range for User Input
      4. Input needs to be transformed to datetime
      5. Filter the dataframe if the dates are not None
      6. Split the dataframe in two, one with review and one without reviews
      7. Rescore the dataframe without review
      8. Show the dataframe with review
      9. Save to database or delete all data from database
      10. Show all the graphs
      '''
      #1. Set Logo of the page
      st.image('pages/d.png', width=150)
      search_bar = st.text_input('Search', placeholder='Search term: "food, service, atmosphere"', key='HI')
      expander_filters = st.sidebar.expander('Filtering Options', expanded=False)
      self.main_c = st.container()
      self.main_c_1, self.main_c_2 = self.main_c.columns(2)
      
      # 1. Get the data
      self.df = self.get_data()
      #. Filter by restaurant name
      restaurants_names = ['*'] + list(lookUpCafe.values())
      emoji_necessary = list_of_emojis[:len(restaurants_names)]          
      restaurant_name = pills("Cafè", restaurants_names, emoji_necessary, key='restaurant_name')
      self.df, self.db_name = self.filter_by_café(restaurant_name, lookUpCafe)

      IsEmpty = self.df.shape[0] == 0
      #st.write(IsEmpty)
      if IsEmpty:
         self.UploadNewData()

      # 2. Search bar
      self.df = self.filter_search_bar(search_bar)
      # 3. Date Range for User Input
      start_date, end_date = expander_filters.date_input('Date Range', [pd.to_datetime(self.df['date_for_filter'].min()),
                                                                        pd.to_datetime(self.df['date_for_filter'].max())],
                                                                        key='date_range')
      self.df, start_date, end_date = self.filter_by_date(self.df, start_date, end_date)

      # 6.1. Split the dataframe in two, one with review and one without reviews
      self.df_with_review = self.df[self.df['Details'] != '']
      self.df_without_review = self.df[self.df['Details'] == '']

      # 7. Rescore the dataframe without review
      self.df_without_review = rescoring_empty(self.df_without_review)

      # 7.1 Filter by keywords
      key_words = expander_filters.multiselect('Keywords', keywords, default = [])
      if key_words != []:
         self.df_with_review = self.df_with_review[self.df_with_review['Keywords'].str.contains('|'.join(key_words), case=False)]

      #7.2 Filter by Day Part
      day_part = expander_filters.multiselect('Day Part', self.df_with_review['Day_Part'].unique().tolist(), default = [])
      if day_part != []:
         self.df_with_review = self.df_with_review[self.df_with_review['Day_Part'].str.contains('|'.join(day_part), case=False)]

      #7.3 Filter by Day of the week
      days_of_the_week = self.df_with_review['Day_Name'].unique().tolist()
      day_of_the_week = expander_filters.multiselect('Day of the week', days_of_the_week, default = [])
      if day_of_the_week != []:
         self.df_with_review = self.df_with_review[self.df_with_review['Day_Name'].str.contains('|'.join(day_of_the_week), case=False)]

      #7.4 Filter by Month
      months = self.df_with_review['Month'].unique().tolist()
      month = expander_filters.multiselect('Month', months, default = [])
      if month != []:
         self.df_with_review = self.df_with_review[self.df_with_review['Month'].str.contains('|'.join(month), case=False)]

      #7.5 Filter by Negative, Neutral, Positive
      sentiment_to_consider = expander_filters.multiselect('Sentiment', ['POSITIVE', 'NEGATIVE', 'neutral'], default = [])
      if sentiment_to_consider != []:
         self.df_with_review = self.df_with_review[self.df_with_review['Sentiment'].str.contains('|'.join(sentiment_to_consider), case=False)]
      
      #7.6 Filter by negative and empty labels
      if expander_filters.toggle('Only Reviews That Needs a Label', value = False, key = 'negative and empty labels'):
         self.df_with_review = self.df_with_review[(self.df_with_review['Sentiment'] == 'NEGATIVE') & (self.df_with_review['Label: Dishoom'] == '')]

      self.to_plot = self.df_with_review
      #st.write(self.db_name)
      name_choosen_db = 'pages/' + restaurant_name + '.db' if restaurant_name != '*' else 'pages/' + self.db_name + '.db'

      if self.name_user == 'AllEars':
         if restaurant_name != '*':
            #st.write(f'Restaurant: {restaurant_name}')
            res_name_db = list(lookUpCafe.keys())[list(lookUpCafe.values()).index(restaurant_name)]
            name_choosen_db = 'pages/' + res_name_db + '.db'
            button_delete_single_res = st.sidebar.button('🗑', use_container_width=True, key = 'delete_single')
            if button_delete_single_res:
                  data_from_db = fetch_data_from_db(name_choosen_db)
                  db = Database_Manager(name_choosen_db)
                  db.delete_all()
                  self.df = self.db_main_manager.get_main_db_from_venue()
                  #st.write(self.df)

         elif restaurant_name == '*': # works fine
            #st.write('ALL RESTAURANTS : * detected')
            button_delete_everything = st.sidebar.button('🗑', use_container_width=True, key = 'delete_all')
            if button_delete_everything:
                  db_main = Database_Manager(self.db_name)
                  db_main.delete_all()
                  st.info('Deleted all data from database')
                  venues = self.df['Reservation: Venue'].unique().tolist()
                  for venue in venues:
                     name_choosen_db = 'pages/' + venue + '.db'
                     db_single_res = Database_Manager(name_choosen_db)
                     db_single_res.delete_all()
                  # I need to delete all the rev from the main db as well
                  st.info('Deleted all data from database')


         if len(self.df_with_review) > 0:
            self.plot()
         try:
            index_to_modify = st.number_input('Review N.', min_value=1, max_value=len(self.df_with_review), value=1, step=1, on_change=None, key=None)
         except:
            st.info('No reviews for this cafe')
            st.stop()

      # CARD
      starts_or_number = st.sidebar.radio('Starts or Number', ['Stars', 'Number'], index=0, key='starts_or_number')
      stars_size = st.slider(
         value = 20,
         min_value = 5,
         max_value = 30,
         step = 1
      )
      import streamlit_antd_components as sac

      with st.form(key='my_form'):
         space_to_save_button = st.empty()
         tab_card, tab_details = st.tabs(['Labels', 'Details'])
         with tab_card:
            row = self.df_with_review.iloc[index_to_modify-1]
            date = row['date_for_filter']
            venue = row['Reservation: Venue']
            time  = row['Reservation: Time']
            day_part = row['Day_Part']
            food = row['Menu Item']
            drink = row['Drink Item']

            is_favorite = row['👍'] == '1'
            is_not_favorite = row['👎'] == '1'
            is_suggestion = row['💡'] == '1'
            label = row['Label: Dishoom']

            c1,c2,c3, c4, c5 = st.columns(5)

            # c1.write(f'**Venue**: {venue}')
            # c2.write(f'**Date**: {date}')
            # c3.write(f'**Time**: {time}')
            # c4.write(f'**Day Part**: {day_part}')
            # c5.write(f'**Suggested to Friend**: {suggestion}')

            # split at the - and get the first part
            # get index from label if there is one
            if label == '' or label == ' ':
               label = []
            else:
               label = label.split('-')
               # strip by removing spaces
               label = [l.strip() for l in label]

            sentiment = row['Sentiment']
            options = ['POSITIVE', 'NEGATIVE', 'neutral']
            index = options.index(sentiment)
            rev_space = st.empty()
            col1, col2, col3,col4 = st.columns(4)

            select_thumbs_up = col1.checkbox('👍', value = is_favorite, key = f't_u {index_to_modify}', help = 'Save as one of the Best Reviews')
            select_thumbs_down = col2.checkbox('👎', value = is_not_favorite, key = f't_d {index_to_modify}', help = 'Save as one of the Worst Reviews')
            select_suggestion = col3.checkbox('💡', value = is_suggestion, key = f't_s {index_to_modify}', help = 'Save as customer Suggestion')
            
            # TRANSLATION ---- 
            if col4.checkbox('Translate to **Eng**', value = False, key = f'translate {index_to_modify}'):
               rev_original = row['Details']
               rev_in_eng = self.translator.translate(rev_original)
               rev = rev_in_eng
               if st.button('Save in English language'):
                  db = Database_Manager(self.db_name)
                  db.modify_details_in_db(rev_original, rev_in_eng)
                  # get restaurant name
                  restaurant_name = row['Reservation: Venue']
                  name_choosen_db = 'pages/' + restaurant_name + '.db'
                  db = Database_Manager(name_choosen_db)
                  db.modify_details_in_db(rev_original, rev_in_eng)
                  all_data = self.walla.classify_review(rev_in_eng)
                  sentiment = all_data[0]
                  confidence = all_data[1]
                  menu_items = all_data[2]
                  keywords_ = all_data[3]
                  drinks_items = all_data[4]
                  db.modify_sentiment_in_db(rev_in_eng, sentiment)
                  db.modify_confidence_in_db(rev_in_eng, confidence)
                  db.modify_food_in_db(rev_in_eng, ' '.join(menu_items))
                  db.modify_keywords_in_db(rev_in_eng, ' '.join(keywords_))
                  db.modify_drink_in_db(rev_in_eng, ' '.join(drinks_items))
                  st.success('Saved')
            #-----------------
            else:
               rev = row['Details']
      
            #st.markdown('''
            #:red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
            #:gray[pretty] :rainbow[colors].''')

            # make the rev red 
            message_markdown = f'**Review**: {rev}'
            food_in = food.split('-')
            food_in = [f for f in food_in if f != '']
            for i, f in enumerate(food_in):
               # make all lower case
               f = f.lower().strip()
               message_markdown = message_markdown.lower()
               message_markdown = message_markdown.replace(f, f':red[{f}]')
            drink_in = drink.split('-')
            drink_in = [d for d in drink_in if d != '']
            for i, d in enumerate(drink_in):
               # make all lower case
               d = d.lower().strip()
               message_markdown = message_markdown.lower()
               message_markdown = message_markdown.replace(d, f':blue[{d}]')
            rev_space.markdown(message_markdown, unsafe_allow_html=True)


            #select_sentiment =  c1.selectbox('Sentiment', ['POSITIVE', 'NEGATIVE', 'neutral'], index = index, key = f'i_{index_to_modify}', help = 'Select the sentiment for the review')
            select_label = st.multiselect('Label', options_for_classification, default = label, key = f'l {rev}', help = 'Select the label for the review')
            all_food = DBMenu().view() # ->(1, Vada Pau), (2, ...)
            all_drinks = DBDrinks().view()
            only_food = [f[1] for f in all_food]
            only_drinks = [d[1] for d in all_drinks]
            select_sentiment = row['Sentiment']

            if food == '' or food == ' ':
               food = []
            else:
               food = food.split('-')
               food = [l.strip() for l in food if l != '']

            # same for drinks
            if drink == '' or drink == ' ':
               drink = []
            else:
               drink = drink.split('-')
               drink = [l.strip() for l in drink if l != '']
               #st.write(drink)
            
            c1,c2 = st.columns(2)
            select_food = c1.multiselect('Food', only_food, default = food, key = f'f {rev}', help = 'Select the food items for the review')
            select_drinks = c2.multiselect('Drinks', only_drinks, default = drink, key = f'd {rev}', help = 'Select the drinks items for the review')

            columns_rating = ['Overall Rating', 'Feedback: Food Rating', 'Feedback: Drink Rating', 'Feedback: Service Rating', 'Feedback: Ambience Rating']
            new_columns_rating = ['New Overall Rating', 'New Food Rating', 'New Drink Rating', 'New Service Rating', 'New Ambience Rating']
            
            columns_for_input = ['Overall', 'Food', 'Drink', 'Service', 'Ambience']
            columns_ = st.columns(len(columns_rating))

            results = []
            for i, col in enumerate(columns_rating):
               value_customer = float(row[col])
               value_new = float(row[new_columns_rating[i]])
               # transform the value into a string
               value_customer = int(value_customer)
               if value_customer == 0:
                  value_customer = 'NAN'
               value_new = int(value_new)
               value_map = {
                           5: 10,
                              4: 9,
                                 3: 8,
                                    2: 5,
                                       1: 1
                           }
               max_val = value_map[value_customer] if value_customer != 'NAN' else 10
               if starts_or_number == 'Number':
                  new_value = columns_[i].number_input(label=f'{columns_for_input[i]} **{value_customer}**', 
                                                       min_value=0, 
                                                       max_value=max_val, 
                                                       value=value_new, 
                                                       step=1, 
                                                       help = 'Select the rating for the review',
                                                       format=None, key=f'rate{i} - {index_to_modify}')
               else:
                  with columns_[i]:
                     new_value = sac.rate(
                        label = f'{columns_for_input[i]} **{value_customer}**',
                        value=value_new, count=max_val, key = f'rate{i} - {index_to_modify}', size = stars_size)

               # add to the list
               results.append(new_value)

            from google_big_query import GoogleBigQuery, TransformationGoogleBigQuery
            def get_sales_date(store_id, date, time = None):
               
               googleconnection = GoogleBigQuery()

               query_for_only_a_date = f'''
               SELECT *,
                  EXTRACT(MONTH FROM DateOfBusiness) AS Month
                  FROM `sql_server_on_rds.Dishoom_dbo_dpvHstCheckSummary`
                  WHERE DateOfBusiness = '{date}'
                        AND FKStoreID IN ({','.join([str(i) for i in store_id])})
               '''
               df = googleconnection.query(query = query_for_only_a_date, as_dataframe = True)
               fig, df = TransformationGoogleBigQuery(df, plot = True).transform()
               # add vertical line on time
               if time is not None:
                  fig.add_vline(x=time, line_width=10, line_color="red", opacity=0.3)
               st.plotly_chart(fig) 

            # venue need to go from the name to the id
            venue_map = {
               'Dishoom Covent Garden': 1,
               'Dishoom Shoreditch': 2,
               'Dishoom Kings Cross': 3,
               'Dishoom Carnaby': 4,
               'Dishoom Edinburgh': 5,
               'Dishoom Kensington': 6,
               'Dishoom Manchester': 7,
               'Dishoom Birmingham': 8,
               'Dishoom Canary Wharf': 9
         }
            
            # get the id from the name
            store_id = venue_map[venue]
            time = time if time != '' else None
            #get_sales_date(store_id= [store_id], date = date, time = time)   

            with st.sidebar.expander('Ratings Scale', expanded=False):
                  st.write('5 = 10')
                  st.write('4 = 9')
                  st.write('3 = 8')
                  st.write('2 = 5')
                  st.write('1 = 1')
            import streamlit_antd_components as sac


            # now we need to save the data to the database
            if space_to_save_button.form_submit_button('Save', use_container_width=True):
                  restaurant_name = row['Reservation: Venue']
                  name_choosen_db = 'pages/' + restaurant_name + '.db'
                  db = Database_Manager(name_choosen_db)

                  db.modify_overall_rating_in_db(rev, results[0])
                  db.modify_food_rating_in_db(rev, results[1])
                  db.modify_drink_rating_in_db(rev, results[2])
                  db.modify_service_rating_in_db(rev, results[3])
                  db.modify_ambience_rating_in_db(rev, results[4])
                  db.modify_sentiment_in_db(rev, select_sentiment)

                  db.modify_food_in_db(rev, '-'.join(select_food))
                  db.modify_drink_in_db(rev, '-'.join(select_drinks))
                  db.modify_label_in_db(rev, '-'.join(select_label))
                  # we can have a max of 3 thumbs up and 3 thumbs down
                  # get restaurant name

                  restaurant_name = row['Reservation: Venue']
                  number_of_thumbs_up_in_res = db.get_number_of_thumbs_up(restaurant_name)
                  number_of_thumbs_down_in_res = db.get_number_of_thumbs_down(restaurant_name)
                  #st.write(f'**Number of thumbs up in {restaurant_name}**: {number_of_thumbs_up_in_res}')
                  #st.write(f'**Number of thumbs down in {restaurant_name}**: {number_of_thumbs_down_in_res}')
                  
                  if number_of_thumbs_down_in_res + 1 > 3 and select_thumbs_down:
                     st.info('You have reached the maximum number of thumbs down for this restaurant')
                     select_thumbs_down = False
                     st.stop()
                  if number_of_thumbs_up_in_res + 1 > 3 and select_thumbs_up:
                     st.info('You have reached the maximum number of thumbs up for this restaurant')
                     select_thumbs_up = False
                     st.stop()

                  db.modify_thumbs_up_in_db(rev, select_thumbs_up)
                  db.modify_thumbs_down_in_db(rev, select_thumbs_down)
                  db.modify_is_suggestion(rev, select_suggestion)
                  st.success('Saved')

            # delete the review
            if st.form_submit_button('Delete', use_container_width=True):
                  if st.button('Confirm and delete', use_container_width=True, type = 'primary'):
                     # get restaurant name
                     restaurant_name = row['Reservation: Venue']
                     name_choosen_db = 'pages/' + restaurant_name + '.db'
                     db = Database_Manager(name_choosen_db)
                     db.delete_review(rev)
                     st.success('Deleted')

         with tab_details:
            try:
               get_sales_date(store_id= [store_id], date = date, time = time)
            except Exception as e:
               st.write(e)
               st.write(self.df_with_review.iloc[index_to_modify-1]['Details'])
               # write all the informations
               st.write(self.df_with_review.iloc[index_to_modify-1])
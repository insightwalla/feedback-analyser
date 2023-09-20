# rewrite as a class Page, clearly separating database connection, UI and processing
from utils import *
from parameters import *
from graphs import *
from database import Database_Manager

def get_data_from_database(db_manager: Database_Manager):
    # 1. get the data from the database
    reviews = db_manager.view()
    df = pd.DataFrame(reviews, columns=['idx'] + db_manager.COLUMNS_FOR_SECTION)
    if not reviews:
        st.warning('No reviews found in the database')
        st.stop()
    return df

def transforming_section_to_check_if_changed(section_df):
    '''This function is used to transform the section_df to check if the data has changed'''
    section_df_copy = section_df.copy()
    # need to go from menu_item to food
    section_df_copy['menu_item'] = section_df_copy['menu_item'].apply(lambda x: clean_food(x))
    section_df_copy['drink_item'] = section_df_copy['drink_item'].apply(lambda x: clean_drinks(x))
    section_df_copy['label'] = section_df_copy['label'].apply(lambda x: clean_label(x))
    # rename menu_item to food, drink_item to drink and details to review
    section_df_copy = section_df_copy.rename(columns={'menu_item': 'food', 'drink_item': 'drink', 'details': 'review'})
    section_df_copy = section_df_copy[['review', 'food', 'drink', 'label']]
    # transform label column to string - separated by space
    section_df_copy['label'] = section_df_copy['label'].apply(lambda x: ' - '.join(x) if x != '' else None)
    return section_df_copy

def get_section_df_and_keywords(df, section):
    if section == 'Product':
        section_df = filter_only_food_related_reviews(df)
        keywords_list = food_keywords
    elif section == 'Service':
        section_df = filter_only_service_related_reviews(df)
        keywords_list = service_keywords
    elif section == 'Ambience':
        keywords_list = key_words_related_to_ambience
        section_df = filter_only_ambience_related_reviews(df)
    return section_df, keywords_list

def save_function(df, data_frame, name = 'pages/reviews.db'):
   '''
   Inside the section ("Product", "Service" or "Ambience") we have the following dataframes:
   - `df`: the dataframe that we get from the database directly
   - `data_frame`: the dataframe that is been loaded from the database, that has the labels, food and drinks

   When the `other_df` is different from the `data_frame`, we need to update the database with the new labels, food and drinks

    1. Prepare the data for the merging, changing the column names, to set the same name for the column that we are going to merge
    2. Actually merging the dataframes
    3. Since we now have two columns with the labels, we need to create a new column with the labels concatenated
    4. Same for the food and drinks
    5. Now we can set the columns names to the original ones
    6. and drop the columns that we don't need anymore
    7. save to database
    
   '''

   #1. Prepare the data for the merging, changing the column names, to set the same name for the column that we are going to merge
   data_frame = data_frame.rename(columns={'review': 'details'})

   #2. Actually merging the dataframes
   new_data = pd.merge(df, data_frame, on='details', how='left')

   #3. Since we now have two columns with the labels, we need to create a new column with the labels concatenated
   new_data['label'] = new_data['label_y'].fillna(new_data['label_x'])
   new_data['label'] = new_data['label'].apply(lambda x: ' - '.join(set(x.split(' - '))))

   #4. Same for the food and drinks
   new_data['menu_item'] = new_data['food'].apply(lambda x: ' - '.join(set(x)) if str(x) != 'nan' else '')
   new_data['drink_item'] = new_data['drink'].apply(lambda x: ' - '.join(set(x)) if str(x) != 'nan' else '')

   #st.write(new_data)
   #st.stop()
   
   new_data = new_data.drop(columns=['label_x', 'label_y', 'food', 'drink'])
   
   # 7. save to database
   save_to_db(new_data, Database_Manager.COLUMNS_FOR_SECTION, name)
   st.success('Saved')


# UI AND CARDS
def UI():
    # 3. create a container for the plot
    container_totals = st.sidebar.container()
    col_image_left, col_right = st.columns(2)
    col_image_left.image('pages/d.png', width=200)
    col_right_graph = col_right.container()

    container_editor = st.container()
    container_index = st.container()
    holder_review = st.empty()
    return col_right_graph, container_totals, container_editor, container_index, holder_review

def card_template(
        card_container : st.container,
        index : int,
        index_review_to_show : int,
        data_frame : pd.DataFrame,
        row : pd.Series,
        ):
    '''
    This card template is used to show the reviews in the UI.
    It runs inside a loop, and it will create a card for each review.
    It will only show the review that is selected by the user.

    Parameters:
    - `card_container`: the container where the card will be placed
    - `index`: the index of the review
    - `index_review_to_show`: the index of the review that will be expanded
    - `data_frame`: the dataframe with the reviews

    
    '''
    rev = row['details']
    food = clean_food(row['menu_item'])
    drink = clean_drinks(row['drink_item'])
    labels = clean_label(row['label'])

    if index == index_review_to_show:
        with card_container.expander(f"Review {index}", expanded=True if index == index_review_to_show else False):
            col1, col2, col3, col4 = st.columns(4)
            col1.write(f"**{row['reservation_venue']}**")
            col2.write(f"**Reservation Date** {row['reservation_date']}" if row['reservation_date'] != '' else f"**Submission Date** {row['date']}") 
            col3.write(f"**Time** {row['time'] if row['time'] != '' else 'Not specified'}")
            col4.write(f'**Suggested to Friends** {row["suggested_to_friend"]}')
            st.write(f"{rev}")


            c1,c2, c3 = st.columns(3)
            labels = c1.multiselect('Label Sentiment', options_for_classification, key=index, default=labels)
            food_selected = c2.multiselect('Label Food', menu_items_lookup, key=f"{index}f", default=food)
            drink_selected = c3.multiselect('Label Drink', drink_items_lookup, key=f"{index}d", default=drink)
            # modify label before adding to the dataframe
            food = food_selected if food_selected != '' else [None]
            drink = drink_selected if drink_selected != '' else [None]

    label = " - ".join(labels) if labels != '' else [None]
    data_frame = pd.concat([data_frame, pd.DataFrame({'review': [rev], 'food': [food], 'drink': [drink], 'label': [label]})], ignore_index=True)

    return data_frame

# MAIN PAGE
class SectionTemplate:
    def __init__(self, name_db = 'pages/reviews.db', section = 'Product'):
        self.name_db = name_db
        self.section = section
        self.df = get_data_from_database(Database_Manager(name_db))
        self.section_df, self.keywords_list = get_section_df_and_keywords(self.df, self.section)
        self.run()

    def run(self):
        col_right_graph, container_totals, container_editor, container_index, card_container = UI()        
        index_review_to_show = container_index.number_input('Review to show', min_value=1, max_value=len(self.section_df), value=1, step=1, on_change=None, key=None)    
        
        data_frame = pd.DataFrame()
        for index, row in self.section_df.iterrows():
            data_frame = card_template(
                                    card_container,
                                    index,
                                    index_review_to_show,
                                    data_frame,
                                    row)
        
        # 5. Complete final database and add index column
        data_frame.index = range(len(data_frame))
        data_frame.index = data_frame.index + 1

        # 5.1 Add the data to the container
        container_editor.experimental_data_editor(data_frame, use_container_width=True)

        # 6. Graphs
        create_pie_chart_completion(data_frame, container_totals)
        create_chart_totals_labels(data_frame, container_totals)
        create_chart_totals_food_and_drinks(data_frame, container_totals, self.section_df)
        creating_keywords_graphs(key_words_list=self.keywords_list, df=self.df, container = col_right_graph)

        create_timeseries_graph_section(self.df, container_totals, col_date= 'date_for_filter')    

        # create the save button
        save_button = st.button('Save')
        if save_button:
            #st.write("Save")
            save_function(self.df, data_frame, self.name_db)

        # Create a checker to save the data if the data has changed
        if not data_frame.equals(transforming_section_to_check_if_changed(self.section_df)):
            #save_button = st.sidebar.button('Save')
            save_function(self.df, data_frame, self.name_db)

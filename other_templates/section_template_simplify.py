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

# MAIN PAGE
class SectionTemplate:
    def __init__(self, name_db = 'pages/details.db', section = 'Product', name_user = ''):
        self.name_db = name_db
        self.section = section
        self.data = get_data_from_database(Database_Manager(name_db))

        # create a dictionary containing {name_venue: pd.DataFrame}
        venues = self.data['reservation_venue'].unique()
        # transform to a list
        venues = venues.tolist()
        venues_dict = {venue: self.data[self.data['reservation_venue'] == venue] for venue in venues}
        
        res_to_rename = {
                'Dishoom Covent Garden': 'D1',
                'Dishoom Shoreditch': 'D2',
                'Dishoom Kings Cross': 'D3',
                'Dishoom Carnaby': 'D4',
                'Dishoom Edinburgh': 'D5',
                'Dishoom Kensington': 'D6',
                'Dishoom Manchester': 'D7',
                'Dishoom Birmingham': 'D8',
                'Dishoom Canary Wharf': 'D9'
            }
        
        restaurants_names = res_to_rename.values()
        restaurants_names = list(restaurants_names)


        selected_restaurant = st.sidebar.selectbox('Select Restaurant', restaurants_names)
        selected_restaurant = [key for key, value in res_to_rename.items() if value == selected_restaurant][0]
        
        st.subheader(f"**{selected_restaurant}**")
        self.name_db_choosen = f'pages/{selected_restaurant}.db'
        self.df_selected = get_data_from_database(Database_Manager(self.name_db_choosen))
        self.section_df, self.keywords_list = get_section_df_and_keywords(self.df_selected, self.section)
        # is the dataframe empty?
        if self.section_df.empty:
            st.warning(f'No reviews relative to {self.section} in {selected_restaurant}')
            st.stop()
        else:
            self.run()

    def run(self):
        col_right_graph, container_totals, container_editor, container_index, card_container = UI()    
        features = ['details', 'menu_item', 'drink_item', 'label']    
        temporary_df = self.section_df[features]
        # change the names of the columns of the dataframe
        temporary_df.columns = ['Review', 'Food', 'Drink', 'Label']
        # transform food drink and label in lists
        temporary_df['Food'] = temporary_df['Food'].apply(lambda x: x.split(' - ') if x != '' else [])
        temporary_df['Drink'] = temporary_df['Drink'].apply(lambda x: x.split(' - ') if x != '' else [])
        temporary_df['Label'] = temporary_df['Label'].apply(lambda x: x.split(' - ') if x != '' else [])
        
        st.dataframe(temporary_df, use_container_width=True)
        # add filter for restaurant 
        index_review_to_show = st.number_input('Review to show', min_value=1, max_value=len(self.section_df), value=1, step=1, on_change=None, key=None)    
        
        # get the review to show
        row = self.section_df.iloc[index_review_to_show-1]
        rev = row['details']
        food = row['menu_item']
        drink = row['drink_item']
        labels = row['label']
        #st.write(f'{type(food)}')
        #st.write(f'{type(drink)}')
        #st.write(f'{type(labels)}')

        food = clean_food(row['menu_item'])
        drink = clean_drinks(row['drink_item'])
        labels = clean_label(row['label'])

        #st.write(f'{type(food)}')
        #st.write(f'{type(drink)}')
        #st.write(f'{type(labels)}')

        # 1. create the UI
        with st.expander('Card', expanded=True):
            c1,c2,c3,c4 = st.columns(4)
            c1.write(f"**{row['reservation_venue']}**")
            c2.write(f"**Reservation Date** {row['reservation_date']}" if row['reservation_date'] != '' else f"**Submission Date** {row['date']}")
            c3.write(f"**Time** {row['time'] if row['time'] != '' else 'Not specified'}")
            c4.write(f'**Suggested to Friends** {row["suggested_to_friend"]}')
            st.write(f"{rev}")

            c1,c2, c3 = st.columns(3)
            labels = c1.multiselect('Label Sentiment', options_for_classification, key=index_review_to_show, default=labels)
            food_selected = c2.multiselect('Label Food', menu_items_lookup, key=f"{index_review_to_show}f", default=food)
            drink_selected = c3.multiselect('Label Drink', drink_items_lookup, key=f"{index_review_to_show}d", default=drink)

            # modify label before adding to the dataframe
            def prepare_food_drink_label(label):
                if len(label) == 0:
                    return ''
                elif len(label) == 1:
                    return f'{label[0]}'
                else:
                    # save as a string
                    return " - ".join(label)
                
                
            food = prepare_food_drink_label(food_selected)
            drink = prepare_food_drink_label(drink_selected)
            label = prepare_food_drink_label(labels)

            #st.write(f"**Label Food** {food}")
            #st.write(f"**Label Drink** {drink}")
            #st.write(f"**Label Sentiment** {label}")

            #st.write(f'{type(food)}')
            #st.write(f'{type(drink)}')
            #st.write(f'{type(label)}')


            # now we need to save the data to the database
            if st.button('Save'):
                db = Database_Manager(self.name_db_choosen)
                db.modify_food_in_db(rev, food)
                db.modify_drink_in_db(rev, drink)
                db.modify_label_in_db(rev, label)
                st.success('Saved')

                
        # 6. Graphs
        create_pie_chart_completion(self.section_df, container_totals)
        create_chart_totals_labels(self.section_df, container_totals)
        create_chart_totals_food_and_drinks(self.section_df, container_totals, self.section_df)
        creating_keywords_graphs(key_words_list=self.keywords_list, df=self.df_selected, container = col_right_graph)
        create_timeseries_graph_section(self.df_selected, container_totals, col_date= 'date_for_filter')    

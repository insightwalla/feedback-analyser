from utils import *
from graphs import *

def get_data_from_database(db_manager: Database_Manager):
    # 1. get the data from the database
    reviews = db_manager.view()
    df = pd.DataFrame(reviews, columns=['idx'] + db_manager.COLUMNS_FOR_SECTION)
    if not reviews:
        st.warning('No reviews found in the database')
        st.stop()
    return df

def final_view(name_db, section = '', name_user = ''):
    '''
    This section will show a view that makes easy to choose the best and worst reviews
    '''
    data = get_data_from_database(Database_Manager('pages/details.db'))

    # create a dictionary containing {name_venue: pd.DataFrame}
    venues = data['reservation_venue'].unique()
    # transform to a list
    venues = venues.tolist()
    venues_dict = {venue: data[data['reservation_venue'] == venue] for venue in venues}
    
    selected_restaurant = st.sidebar.selectbox('Select Restaurant', list(venues_dict.keys()))
    st.subheader(f"**{selected_restaurant}**")
    name_db_choosen = f'pages/{selected_restaurant}.db'
    df = get_data_from_database(Database_Manager(name_db_choosen))
    
    # 1. get the data from the database
    #df = pd.DataFrame(reviews, columns=['idx'] + Database_Manager.COLUMNS_FOR_SECTION)

    c1,c2 = st.columns(2)
    expander_best = c1.empty()
    expander_worst = c2.empty()
    expander_light = st.empty()

    cols_worst = ['💡','👎', 'details', 'sentiment', 'label', 'confidence', 'overall_rating', 'food_rating', 'drink_rating', 'service_rating', 'ambience_rating', 'suggested_to_friend', 'keywords']
    cols_best = ['💡','👍', 'details', 'sentiment', 'label', 'confidence', 'overall_rating', 'food_rating', 'drink_rating', 'service_rating', 'ambience_rating', 'suggested_to_friend', 'keywords']
    # 2. get the worst reviews
    worst_reviews = get_worst_reviews(df)
    # change the best_positive and worst_negative to True or False
    worst_reviews['👍'] = worst_reviews['👍'].apply(lambda x: True if str(x) == '1' else False)
    worst_reviews['👎'] = worst_reviews['👎'].apply(lambda x: True if str(x) == '1' else False)
    worst_reviews['💡'] = worst_reviews['💡'].apply(lambda x: True if str(x) == '1' else False)
    
    # 3. get the best reviews
    best_reviews = get_best_reviews(df)
    # change the best_positive and worst_negative to True or False
    best_reviews['👍'] = best_reviews['👍'].apply(lambda x: True if str(x) == '1' else False)
    best_reviews['👎'] = best_reviews['👎'].apply(lambda x: True if str(x) == '1' else False)
    best_reviews['💡'] = best_reviews['💡'].apply(lambda x: True if str(x) == '1' else False)
    
    # change column names to thumbs up and thumbs down
    c_1, c_2 = st.columns(2)
    # add subheaders
    c_2.subheader('👎 Worst reviews')
    c_1.subheader('👍 Best reviews')

    worst_edit = c_2.data_editor(worst_reviews[cols_worst])
    best_edit = c_1.data_editor(best_reviews[cols_best])

    with expander_best:
        best = best_edit[best_edit['👍'] == True]
        with st.expander(f'👍 Best reviews ({best.shape[0]}/3)', expanded=False):
            best = best['details'].to_list()
            # get all row from the original dataframe with the detalils
            st.write('---')
            columns_to_show = ['source', 'date_for_filter', 'day_part', 'day_name', 'time', 'menu_item', 'drink_item']
            columns_to_rename = ['Source', 'Date', 'Day part', 'Day name', 'Time', 'Food', 'Drink']
            for i, review in enumerate(best):
                # get all row from the original dataframe with the detalils
                row = df[df['details'] == review]
                row = row[columns_to_show]
                row.columns = columns_to_rename
                st.write(row)
                # get food and drinks and 
                st.write(f'{i+1}.')
                st.write(review)
                st.write('---')

    with expander_worst:
        worst = worst_edit[worst_edit['👎'] == True]
        with st.expander(f' 👎 Worst reviews ({worst.shape[0]}/3)', expanded=False):
            # same as above
            worst = worst['details'].to_list()
            columns_to_show = ['reservation_venue', 'date_for_filter', 'day_part', 'day_name', 'time', 'menu_item', 'drink_item']
            columns_to_rename = ['reservation_venue', 'Date', 'Day part', 'Day name', 'Time', 'Food', 'Drink']
            st.write('---')

            for i, review in enumerate(worst):
                row = df[df['details'] == review]
                row = row[columns_to_show]
                row.columns = columns_to_rename
                st.write(row)
                st.write(f'{i+1}.')
                st.write(review)
                st.write('---')

    with expander_light:
        worst_sugg = worst_edit[worst_edit['💡'] == True]
        best_sugg = best_edit[best_edit['💡'] == True]
        # concatenate the two dataframes
        sugg = pd.concat([worst_sugg, best_sugg])
        with st.expander(f'💡 Suggestions from Customers ({sugg.shape[0]})', expanded=False):

            for i, review in enumerate(sugg['details'].to_list()):
                st.write(f'{i+1}.')
                st.write(review)
                st.write('---')
                
    with st.expander('Stats'):
        # get total reviews
        tot_rev = df.shape[0]
        # tot negative reviews
        tot_neg = df[df['sentiment'] == 'NEGATIVE'].shape[0]
        # tot positive reviews
        tot_pos = df[df['sentiment'] == 'POSITIVE'].shape[0]
        # tot neutral reviews
        tot_neu = df[df['sentiment'] == 'NEUTRAL'].shape[0]


        # get total with label different from "" (meaning labelled)
        total_labelled = df[df['label'] != ''].shape[0]
        # get total with label "" (meaning unlabelled)
        total_unlabelled = df[df['label'] == '']
        # get total with label "" and sentiment NEGATIVE (meaning unlabelled and negative)
        total_unlabelled = total_unlabelled[total_unlabelled['sentiment'] == 'NEGATIVE']
        # get the actual reviews that need to be labelled
        total_unlabelled_rev = total_unlabelled['details'].tolist() 
        # get the number of unlabelled reviews
        total_unlabelled = total_unlabelled.shape[0]

        st.write('Total reviews: ', tot_rev)
        st.write('Total negative reviews: ', tot_neg)
        st.write('Total positive reviews: ', tot_pos)
        st.write('Total neutral reviews: ', tot_neu)
        st.write('Total labelled reviews: ', total_labelled)
        st.write('Total unlabelled reviews: ', total_unlabelled)
        st.write('Total unlabelled reviews: ', pd.DataFrame(total_unlabelled_rev))

    # merge the two dataframes
    df_new = pd.concat([best_edit, worst_edit])
    # keep only details and thumbs up and down and suggestions
    df_new = df_new[['details', '👍', '👎', '💡']]
    # add false in nan
    df_new = df_new.fillna(False)
    # merge df and df new at details
    df = df.merge(df_new, on='details', how='left')
    # drop all the columns that are not needed (thumbs up and down and suggestions)
    df = df.drop(columns=['👍_x', '👎_x', '💡_x'])
    # rename the columns
    df = df.rename(columns={'👍_y': '👍', '👎_y': '👎', '💡_y': '💡'})
    df = df[Database_Manager.COLUMNS_FOR_SECTION]
    #st.write(df)
    if st.sidebar.button('Save'):
        save_to_db(df = df, cols= Database_Manager.COLUMNS_FOR_SECTION, name=name_db_choosen)
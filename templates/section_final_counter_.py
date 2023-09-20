from utils import *
from graphs import *

def final_page(name_db: str, section: str, name_user: str):
    st.write('Final page')

    # need to get the data for each restaurant
    data = Database_Manager(name_db).get_main_db_from_venue()

    # get unique venues
    list_of_venue = data['Reservation: Venue'].unique()
    # for each venue get the ones with negative feedback
    for i, venue in enumerate(list_of_venue):
        venue_data = data[data['Reservation: Venue'] == venue]
        venue_data_to_lab = venue_data[venue_data['Sentiment'] == 'NEGATIVE']
        tot_ = len(venue_data_to_lab) + 6
        tot_done = len(venue_data_to_lab[venue_data_to_lab['Label: Dishoom'] != ''])
        tot_not_done = len(venue_data_to_lab[venue_data_to_lab['Label: Dishoom'] == ''])
        tot_done_before = len(venue_data_to_lab[venue_data_to_lab['Label: Dishoom'] == 'Done'])
        # get total thumbs up and thumbs down
        # thumbs up emoji is  👍
        # thumbs down emoji is 👎
        thumbs_up = venue_data[venue_data['👍'] == '1']
        thumbs_down = venue_data[venue_data['👎'] == '1']

        # add the number to the total
        tot_done += len(thumbs_up) + len(thumbs_down)

        # get suggestions 
        suggestions = venue_data[venue_data['💡'] == '1']
        number_of_thumbs_up = len(thumbs_up)
        number_of_thumbs_down = len(thumbs_down)
        
        
        try:
            message = venue + f' **{round(tot_done/tot_*100, 2)}%**' 
        except:
            message = venue
        
        with st.expander(message):
            tot_already_done = len(venue_data_to_lab[venue_data_to_lab['Label: Dishoom'] != ''])
            tab_pie, tab_good, tab_bad, tab_sugg = st.tabs([f'Reviews {len(venue_data_to_lab)}/{tot_already_done}',
                                                            f'Good {number_of_thumbs_up}/3',
                                                            f'Bad {number_of_thumbs_down}/3',
                                                            f'Suggestions {len(suggestions)}'])

            with tab_pie:
                # now create a pie chart
                fig = go.Figure(data=[go.Pie(labels=['Done', 'Not Done'], values=[tot_done, tot_not_done])])
                fig.update_layout(title_text=venue)
                # green for done, red for not done
                fig.update_traces(marker_colors=['green', 'red'])
                # set opacity
                fig.update_traces(opacity=0.6, textinfo='percent+label')
                # set size 200x200
                st.plotly_chart(fig, use_container_width=True)


            with tab_good:
                for good in thumbs_up['Details'].tolist():
                    st.write(good)

            with tab_bad:
                for bad in thumbs_down['Details'].tolist():
                    st.write(bad)

            with tab_sugg:
                for sugg in suggestions['Details'].tolist():
                    st.write(sugg)

    # add a complete df 
    with st.expander('View all data'):
        data = data.rename(columns={'Overall Rating': 'Overall', 'Feedback: Food Rating': 'Food', 'Feedback: Service Rating': 'Service', 'Feedback: Ambience Rating': 'Ambience',
                                    'Feedback: Drink Rating': 'Drink'})
        st.write(data)

    # create a download link
    def get_table_download_link(data):
        # rename the columns that have emoji
        data = data.rename(columns={'👍': 'thumbs_up', '👎': 'thumbs_down', '💡': 'suggestions'})
        # create a link to download the dataframe
        csv = data.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()  # some strings <-> bytes conversions necessary here
        href = f'<a href="data:file/csv;base64,{b64}" download="data.csv">Download csv file</a>'
        return href


    st.markdown(get_table_download_link(data), unsafe_allow_html=True)

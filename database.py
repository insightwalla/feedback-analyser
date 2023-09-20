import sqlite3
import streamlit as st
import pandas as pd
#from parameters import menu_items_lookup, drink_items_lookup, keywords, service_keywords, key_words_related_to_ambience

service_keywords = [
      'service', 'waiter', 'waitress', 'waiters', 'waitresses', 'waitstaff',
      'host', 'hostess', 'hostesses', 'hosts', 'server', 'servers', 'staff',
      'price', 'bill', 'rush', 'rushing', 'rushes', 'rushed', 'rush','birthday',
      'pricing', 'priced', 'pricey', 'expensive', 'expensiveness', 'expensiveness',
   ]

key_words_related_to_ambience = ['atmosphere', 'ambience', 'chairs', 'tables', 'seating',
                                       'decor', 'interior', 'design', 'lighting', 'music', 'sound',
                                       'noise', 'noisy', 'quiet', 'loud', 'crowded', 'crowd', 'crowds',
                                       'crowding', 'clean', 'peaceful', 'toilets','parking','toilet',
                                       
]
   
food_keywords = [
      'cocktails', 'chilli', 'chutney', 'chutneys',
      'icecream', 'ice-cream', 'ice cream', 'icecreams', 'ice-creams', 'ice creams',
      'food', 'taste', 'tastes', 'flavor', 'flavors', 'dish', 'dishes', 'menu',
      'meal', 'meals', 'portions', 'portion', 'portioned', 'portioning', 'portioned',
      'quality', 'quantity', 'quantities', 'quantitied', 'quantitied', 'quantitied', 'Biryani']

keywords = service_keywords + key_words_related_to_ambience + food_keywords

menu_items_lookup = [
         'Akuri',
         'B & E Naan',
         'Bacon Naan',
         'Basmati Kheer',
         'Basmati Rice',
         'Bhel',
         'Big Bombay',
         'Black Daal',
         'Bowl of Greens',
         'BS Keema Pau',
         'BS Pau Bhaji',
         'Butter Corn OTC',
         'Butter Paanch',
         'Calamari',
         'Ch. Chawal (V)',
         'Ch. Puri(V)',
         'Chck Tikka Roll',
         'Cheese Naan',
         'Chicken Biryani',
         'Chicken Ruby',
         'Chicken Tikka',
         'Chilli Br Salad',
         'Chilli Chicken',
         'Choc Pud w/chil',
         'Choc Pud w/Cinn',
         'Chocolate Chai',
         'Chole Puri',
         'Dbl Bacon Naan',
         'Egg Naan',
         'Garlic Naan',
         'Gulab Jamun',
         'GunpowderPotato',
         'House Chaat',
         'Jackfruit Birya',
         'Kachumber',
         'Keema Per Eedu',
         'Kejriwal',
         'Khichia WChundo',
         'Kulfi Malai',
         'Kulfi Mango',
         'Kulfi Pistach.',
         'Lamb Chops',
         'Lamb Raan',
         'Lamb Raan Bun',
         'Lamb Samosas',
         'Mango& Fenn',
         'Masala Prawns',
         'Mattar Paneer',
         'Murgh Malai',
         'Mutton Chp Korm',
         'Mutton Pepper F',
         'Nihari',
         'Nihari Biryani',
         'Okra Fries',
         'Paneer Roll',
         'Paneer Tikka',
         'Parsi Omelette',
         'Pineapple Tikka',
         'Plain Naan',
         'Prawn Koliwada',
         'Raita',
         'Roomali Roti',
         'S & E Naan',
         'Saus Naan',
         'Sheekh Kabab',
         'Side Beans',
         'Tandoor  Roti',
         'Uttapam Stack',
         'Vada Pau',
         'Veg. Samosas',
         'Vegan Bombay',
         'Vegan Saus Naan',
         'Wrestler Naan',
         'Xmas Feast',
         'CCT',
      ]

drink_items_lookup = [
         'Beer',
         'Colaba Colada',
         'Mojito',
         'Mango Lassi',
         'Breakfast Lassi',
         'Coffee',
         'Americano',
         'Espresso',
         'Latte',
         '33cl Kingfisher',
         '65cl Kingfisher',
         'Aflatoon',
         'Amb. Viog Btl',
         'Bob Sauv Blanc',
         'Bolly Bellini',
         'Quinde Gls',
         'Quinde Malbec',
         'Sali Boti Full',
         'Sali Boti Half',
         'Sonia Negroni',
         'Sparkling Water',
         'Spumante Btl',
         'Spumante Gls',
         'Still Water',
         'Viceroy OF',
         'Watermelon Shbt',
         'Syrah Punctum',
         'Kohinoor Fizz',
         'Meenakshi Moj',
         'Nimbu Soda',
         'Oat Chai',
         'P. Grig Btl',
         'P. Grigio Gls',
         'Passionfr Shbt',
         'Peacock Cider',
         'Grenache Btl',
         'Grenache Gls',
         'Chai',
         'handra Bose',
         'Coloba Colada',
         'Cyrus Irani',
         'Dhoble',
         'Diet Coke 33cl',
         'Dish Esp Mtini',
         'Dishoom IPA',
         'East IndiaGim',
         'Folle Grillo Bt',
         'FolleGrillo Gls',
         'Fresh Orange J']

class DBMenu:
    def __init__(self):
        self.conn = sqlite3.connect('menu.db')
        self.cur = self.conn.cursor()
        # create a table called food
        # check if the table exists
        if self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='food'").fetchone() is None:
            self.cur.execute("CREATE TABLE IF NOT EXISTS food (id INTEGER PRIMARY KEY, name text)")
            self.populate_db(menu_items_lookup)

    def populate_db(self, menu_items_lookup):
        '''Populate the database with menu items'''
        for item in menu_items_lookup:
            self.cur.execute("INSERT INTO food VALUES (NULL, ?)", (item,))
            self.conn.commit()

    def view(self, as_df=False):
        '''Returns all the rows from the database'''
        self.cur.execute("SELECT * FROM food")
        rows = self.cur.fetchall()
        if as_df:
            return pd.DataFrame(rows, columns=['id', 'name'])
        else:
            return rows
        
    def delete_all(self):
        self.cur.execute("DELETE FROM food")
        self.conn.commit()

    def save_all(self, list_of_menu_items):
        # delete all and insert all
        self.delete_all()
        self.populate_db(list_of_menu_items)

# same for drinks
class DBDrinks:
    def __init__(self):
        self.conn = sqlite3.connect('menu.db')
        self.cur = self.conn.cursor()
        # create a table called food
        # check if the table exists
        if self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='drinks'").fetchone() is None:
            self.cur.execute("CREATE TABLE IF NOT EXISTS drinks (id INTEGER PRIMARY KEY, name text)")
            self.populate_db(drink_items_lookup)

    def populate_db(self, drink_items_lookup):
        '''Populate the database with menu items'''
        for item in drink_items_lookup:
            self.cur.execute("INSERT INTO drinks VALUES (NULL, ?)", (item,))
            self.conn.commit()

    def view(self, as_df=False):
        '''Returns all the rows from the database'''
        self.cur.execute("SELECT * FROM drinks")
        rows = self.cur.fetchall()
        if as_df:
            return pd.DataFrame(rows, columns=['id', 'name'])
        else:
            return rows
        
    def delete_all(self):
        self.cur.execute("DELETE FROM drinks")
        self.conn.commit()

    def save_all(self, list_of_drink_items):
        # delete all and insert all
        self.delete_all()
        self.populate_db(list_of_drink_items)

# same for list of product related keywords
class DBProductKeywords:
    def __init__(self):
        self.conn = sqlite3.connect('menu.db')
        self.cur = self.conn.cursor()
        # create a table called food
        # check if the table exists
        if self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='product_keywords'").fetchone() is None:
            self.cur.execute("CREATE TABLE IF NOT EXISTS product_keywords (id INTEGER PRIMARY KEY, name text)")
            self.populate_db(keywords)

    def populate_db(self, keywords):
        '''Populate the database with menu items'''
        for item in keywords:
            self.cur.execute("INSERT INTO product_keywords VALUES (NULL, ?)", (item,))
            self.conn.commit()

    def view(self, as_df=False):
        '''Returns all the rows from the database'''
        self.cur.execute("SELECT * FROM product_keywords")
        rows = self.cur.fetchall()
        if as_df:
            return pd.DataFrame(rows, columns=['id', 'name'])
        else:
            return rows
    
    def delete_all(self):
        self.cur.execute("DELETE FROM product_keywords")
        self.conn.commit()

    def save_all(self, list_of_product_keywords):
        # delete all and insert all
        self.delete_all()
        self.populate_db(list_of_product_keywords)

# same for Service and Ambience
class DBServiceKeywords:
    def __init__(self):
        self.conn = sqlite3.connect('menu.db')
        self.cur = self.conn.cursor()
        # create a table called food
        # check if the table exists
        if self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='service_keywords'").fetchone() is None:
            self.cur.execute("CREATE TABLE IF NOT EXISTS service_keywords (id INTEGER PRIMARY KEY, name text)")
            self.populate_db(service_keywords)

    def populate_db(self, service_keywords):
        '''Populate the database with menu items'''
        for item in service_keywords:
            self.cur.execute("INSERT INTO service_keywords VALUES (NULL, ?)", (item,))
            self.conn.commit()

    def view(self, as_df=False):
        '''Returns all the rows from the database'''
        self.cur.execute("SELECT * FROM service_keywords")
        rows = self.cur.fetchall()
        if as_df:
            return pd.DataFrame(rows, columns=['id', 'name'])
        else:
            return rows
        
    def delete_all(self):
        self.cur.execute("DELETE FROM service_keywords")
        self.conn.commit()

    def save_all(self, list_of_service_keywords):
        # delete all and insert all
        self.delete_all()
        self.populate_db(list_of_service_keywords)

# same for Ambience
class DBAmbienceKeywords:
    def __init__(self):
        self.conn = sqlite3.connect('menu.db')
        self.cur = self.conn.cursor()
        # create a table called food
        # check if the table exists
        if self.cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ambience_keywords'").fetchone() is None:
            self.cur.execute("CREATE TABLE IF NOT EXISTS ambience_keywords (id INTEGER PRIMARY KEY, name text)")
            self.populate_db(key_words_related_to_ambience)

    def populate_db(self, key_words_related_to_ambience):
        '''Populate the database with menu items'''
        for item in key_words_related_to_ambience:
            self.cur.execute("INSERT INTO ambience_keywords VALUES (NULL, ?)", (item,))
            self.conn.commit()

    def view(self, as_df=False):
        '''Returns all the rows from the database'''
        self.cur.execute("SELECT * FROM ambience_keywords")
        rows = self.cur.fetchall()
        if as_df:
            return pd.DataFrame(rows, columns=['id', 'name'])
        else:
            return rows
        
    def delete_all(self):
        self.cur.execute("DELETE FROM ambience_keywords")
        self.conn.commit()

    def save_all(self, list_of_ambience_keywords):
        # delete all and insert all
        self.delete_all()
        self.populate_db(list_of_ambience_keywords)
    
class Database_Manager:
   COLUMNS_FOR_SECTION = [
      'details', 'sentiment', 'confidence', 'menu_item',
      'overall_rating', 'food_rating', 'drink_rating',
      'service_rating', 'ambience_rating',
      'date', 'time', 'reservation_date',
      'keywords', 'drink_item', 'reservation_venue', 'label',
      'week', 'month', 'day_name', 'day_part', 'suggested_to_friend', '👍', '👎',
      'source', 'year', 'week_yr', 'month_yr', 'date_for_filter', "💡",
        # add 5 new columns for the New Scores
            'new_overall_rating', 'new_food_rating', 'new_drink_rating',
            'new_service_rating', 'new_ambience_rating'
      ]

   COLUMNS_FOR_CREATION = [
      'Details', 'Sentiment', 'Confidence', 'Menu Item', 
      'Overall Rating','Feedback: Food Rating', 'Feedback: Drink Rating',
      'Feedback: Service Rating', 'Feedback: Ambience Rating',
      'Date Submitted', 'Reservation: Time', 'Reservation: Date',
      'Keywords', 'Drink Item', 'Reservation: Venue', 'Label: Dishoom', 
      'Week', 'Month', 'Day_Name', 'Day_Part', 'Suggested to Friend', '👍', '👎',
      'Source', 'Year', 'Week_Year', 'Month_Year', 'date_for_filter', "💡",
      # add 5 new columns for the New Scores
        'New Overall Rating', 'New Food Rating', 'New Drink Rating',
        'New Service Rating', 'New Ambience Rating'
      ]
   
   def __init__(self, db):
      self.conn = sqlite3.connect(db)
      self.cur = self.conn.cursor()
      query = ",".join([col + ' text' for col in self.COLUMNS_FOR_SECTION])

      self.cur.execute(f"""
                        CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, 
                        """ + query + ')')

      self.conn.commit()

   def view(self):
      '''Returns all the rows from the database'''
      self.cur.execute("SELECT * FROM reviews")
      rows = self.cur.fetchall()
      return rows
   
   def delete_all(self):
      self.cur.execute("DELETE FROM reviews")
      self.conn.commit()

   def delete_Table(self):
      self.cur.execute("DROP TABLE reviews")
      self.conn.commit()

   def insert(self, *row):
      self.cur.execute(f"INSERT INTO reviews VALUES (NULL, {','.join(['?' for i in range(len(row))])})", (row))
      self.conn.commit()
      
   def insert_multiple(self, rows):
      self.cur.executemany(f"INSERT INTO reviews VALUES (NULL, {','.join(['?' for i in range(len(self.COLUMNS_FOR_SECTION))])})", rows)
      self.conn.commit()
      # close the connection
      self.conn.close()

   def insert_multiple_with_id(self, rows):
        try:
            self.cur.executemany(f"INSERT INTO reviews VALUES ({','.join(['?' for i in range(len(self.COLUMNS_FOR_SECTION) + 1)])})", rows)
            self.conn.commit()
            # close the connection
            self.conn.close()
        except:
            self.cur.executemany(f"INSERT INTO reviews VALUES ({','.join(['?' for i in range(len(self.COLUMNS_FOR_SECTION) + 1)])})", rows)
            self.conn.commit()
            # close the connection
            self.conn.close()

   def run_query(self, query):
      self.cur.execute(query)
      self.conn.commit()
      return self.cur.fetchall()
   
   def create_database_for_each_venue(self):
      # get all data 
      list_of_venue = self.run_query("SELECT DISTINCT reservation_venue FROM reviews")
      #st.write(list_of_venue)
      # get all the data for each venue
      for venue in list_of_venue:
         venue = venue[0]
         data = self.run_query(f"SELECT * FROM reviews WHERE reservation_venue = '{venue}'")
         # create a new database for each venue
         db = Database_Manager(f'pages/{venue}.db')
         # delete all the data
         db.delete_all()
         db.insert_multiple_with_id(data)
         db.conn.close()
      #st.stop()

   def get_main_db_from_venue(self):
      # from each venue, get all the data
      list_of_venue = self.run_query("SELECT DISTINCT reservation_venue FROM reviews")
      # create a new database for each venue
      data = []
      for venue in list_of_venue:
         venue = venue[0]
         db = Database_Manager(f'pages/{venue}.db')
         data.append(pd.DataFrame(db.view()))
         db.conn.close()
      # insert all the data into the main database
      # transform into dataframe
      data = pd.concat(data)
      data.columns = ['idx'] + self.COLUMNS_FOR_CREATION
      #st.write(data)
      return data

   def modify_food_in_db(self, review, food):
      sql = "UPDATE reviews SET menu_item = ? WHERE details = ?"
      self.cur.execute(sql, (food, review))
      self.conn.commit()

   def modify_drink_in_db(self, review, drink):
      sql = "UPDATE reviews SET drink_item = ? WHERE details = ?"
      self.cur.execute(sql, (drink, review))
      self.conn.commit()

   def modify_label_in_db(self, review, label):
      sql = "UPDATE reviews SET label = ? WHERE details = ?"
      self.cur.execute(sql, (label, review))
      self.conn.commit()

   # last modify

   def modify_overall_rating_in_db(self, review, rating):
      sql = "UPDATE reviews SET new_overall_rating = ? WHERE details = ?"
      self.cur.execute(sql, (rating, review))
      self.conn.commit()

   def modify_food_rating_in_db(self, review, rating):

      sql = "UPDATE reviews SET new_food_rating = ? WHERE details = ?"
      self.cur.execute(sql, (rating, review))
      self.conn.commit()

   def modify_drink_rating_in_db(self, review, rating):

      sql = "UPDATE reviews SET new_drink_rating = ? WHERE details = ?"
      self.cur.execute(sql, (rating, review))
      self.conn.commit()

   def modify_service_rating_in_db(self, review, rating):
         
         sql = "UPDATE reviews SET new_service_rating = ? WHERE details = ?"
         self.cur.execute(sql, (rating, review))
         self.conn.commit()

   def modify_ambience_rating_in_db(self, review, rating):

      sql = "UPDATE reviews SET new_ambience_rating = ? WHERE details = ?"
      self.cur.execute(sql, (rating, review))
      self.conn.commit()

   def modify_sentiment_in_db(self, review, sentiment):
         
         sql = "UPDATE reviews SET sentiment = ? WHERE details = ?"
         self.cur.execute(sql, (sentiment, review))
         self.conn.commit()   

   def modify_thumbs_up_in_db(self, review, thumbs_up):
         
         sql = "UPDATE reviews SET 👍 = ? WHERE details = ?"
         self.cur.execute(sql, (thumbs_up, review))
         self.conn.commit()

   def modify_thumbs_down_in_db(self, review, thumbs_down):
            
            sql = "UPDATE reviews SET 👎 = ? WHERE details = ?"
            self.cur.execute(sql, (thumbs_down, review))
            self.conn.commit()   


   def modify_is_suggestion(self, review, is_suggestion):
         
         sql = "UPDATE reviews SET 💡 = ? WHERE details = ?"
         self.cur.execute(sql, (is_suggestion, review))
         self.conn.commit()

   def modify_details_in_db(self, original_review, review_new):
         
         sql = "UPDATE reviews SET details = ? WHERE details = ?"
         self.cur.execute(sql, (review_new, original_review))
         self.conn.commit()

   def modify_confidence_in_db(self, review, confidence):
            sql = "UPDATE reviews SET confidence = ? WHERE details = ?"
            self.cur.execute(sql, (confidence, review))
            self.conn.commit()

   def modify_keywords_in_db(self, review, keywords):
            sql = "UPDATE reviews SET keywords = ? WHERE details = ?"
            self.cur.execute(sql, (keywords, review))
            self.conn.commit()
   def get_number_of_thumbs_up(self, restaurant):
      return self.run_query(f"SELECT COUNT(👍) FROM reviews WHERE 👍 = '1' AND reservation_venue = '{restaurant}'")[0][0]
   
   def get_number_of_thumbs_down(self, restaurant):
      return self.run_query(f"SELECT COUNT(👎) FROM reviews WHERE 👎 = '1' AND reservation_venue = '{restaurant}'")[0][0]
   
   def delete_review(self, review):
      sql = "DELETE FROM reviews WHERE details = ?"
      self.cur.execute(sql, (review,))
      self.conn.commit()
    
   def delete_reviews_from_venue(self, venue):
    sql = "DELETE FROM reviews WHERE reservation_venue = ?"
    self.cur.execute(sql, (venue,))
    self.conn.commit()
    
if __name__ == "__main__":
   db = Database_Manager('/Users/robertoscalas/Desktop/demo_working_version/pages/reviews.db')
   db.delete_Table()

import numpy as np
import pandas as pd
import time as tm
from datetime import datetime

categories = ['Lifestyle',
              'Arts_and_Entertainment',
              'Puzzles_and_Brainteasers',
              'Motorcycles',
              'Beauty_and_Cosmetics',
              'Geriatric_and_Aging_Care',
              'Unknown',
              'Visual_Arts_and_Design',
              'Gardening',
              'Accommodation_and_Hotels',
              'Home_Improvement_and_Maintenance',
              'Jobs_and_Employment',
              'Animals',
              'Board_and_Card_Games',
              'Search_Engines',
              'Ground_Transportation',
              'Tennis',
              'Basketball',
              'Fantasy_Sports',
              'Mens_Health',
              'Bingo',
              'Boats',
              'Climbing',
              'Coupons_and_Rebates',
              'Email',
              'Rugby',
              'Energy_Industry',
              'Holidays_and_Seasonal_Events',
              'Programming_and_Developer_Software',
              'Biotechnology_and_Pharmaceuticals',
              'News_and_Media',
              'Business_Services',
              'Business_and_Consumer_Services',
              'Banking_Credit_and_Lending',
              'Heavy_Industry_and_Engineering',
              'Philosophy',
              'Philanthropy',
              'Public_Health_and_Safety',
              'Romance_and_Relationships',
              'Price_Comparison',
              'Social_Sciences',
              'Interior_Design',
              'Accounting_and_Auditing',
              'Aerospace_and_Defense',
              'Vegetarian_and_Vegan',
              'Roleplaying_Games',
              'Vehicles',
              'Casinos',
              'Graphics_Multimedia_and_Web_Design',
              'Poker',
              'Fashion_and_Apparel',
              'Computers_Electronics_and_Technology',
              'Marketplace',
              'Cycling_and_Biking',
              'Pet_Food_and_Supplies',
              'Fishing',
              'Chemical_Industry',
              'Animation_and_Comics',
              'Tobacco',
              'Furniture',
              'Sports_Betting',
              'Tickets',
              'Alternative_and_Natural_Medicine',
              'Relocation_and_Household_Moving',
              'Reference_Materials',
              'Insurance',
              'Publishing_and_Printing',
              'Humor',
              'Medicine',
              'Aviation',
              'Restaurants_and_Delivery',
              'Lottery',
              'Libraries_and_Museums',
              'Pharmacy',
              'E-commerce_and_Shopping',
              'Architecture',
              'Literature',
              'Human_Resources',
              'Books_and_Literature',
              'Childcare',
              'Faith_and_Beliefs',
              'Womens_Health',
              'Mental_Health',
              'Biology',
              'Textiles',
              'Sports',
              'Community_and_Society',
              'Beverages',
              'Immigration_and_Visas',
              'Volleyball',
              'Shipping_and_Logistics',
              'Developmental_and_Physical_Disabilities',
              'Adult',
              'Earth_Sciences',
              'Photography',
              'Golf',
              'Hobbies_and_Leisure',
              'Computer_Hardware',
              'Motorsports',
              'Antiques_and_Collectibles',
              'Cooking_and_Recipes',
              'Camping_Scouting_and_Outdoors',
              'Law_and_Government',
              'Health_Conditions_and_Concerns',
              'Music',
              'Health',
              'File_Sharing_and_Hosting',
              'Computer_Security',
              'Nutrition_Diets_and_Fitness',
              'LGBTQ',
              'Weddings',
              'Boxing',
              'Classifieds',
              'Air_Travel',
              'National_Security',
              'Gambling',
              'Jewelry_and_Luxury_Products',
              'Travel_and_Tourism',
              'Models',
              'Education',
              'TV_Movies_and_Streaming',
              'Home_and_Garden',
              'Auctions',
              'Gifts_and_Flowers',
              'Grants_Scholarships_and_Financial_Aid',
              'Online_Marketing',
              'Government',
              'Pets_and_Animals',
              'Martial_Arts',
              'Crafts',
              'Running',
              'Transportation_and_Excursions',
              'Childrens_Health',
              'Math',
              'Dictionaries_and_Encyclopedias',
              'Consumer_Electronics',
              'Construction_and_Maintenance',
              'Metals_and_Mining',
              'Addictions',
              'Extreme_Sports',
              'American_Football',
              'Physics',
              'Makes_and_Models',
              'Environmental_Science',
              'Food_and_Drink',
              'Maps',
              'Real_Estate',
              'Birds',
              'Agriculture',
              'Science_and_Education',
              'Fish_and_Aquaria',
              'Web_Hosting_and_Domain_Names',
              'Decease',
              'Business_Training',
              'Finance',
              'Hunting_and_Shooting',
              '',
              'Pets',
              'Dentist_and_Dental_Services',
              'Chemistry',
              'Ancestry_and_Genealogy',
              'Advertising_Networks',
              'Public_Records_and_Directories',
              'Video_Games_Consoles_and_Accessories',
              'Waste_Water_and_Environmental',
              'Soccer',
              'Car_Rentals',
              'Performing_Arts',
              'Social_Networks_and_Online_Communities',
              'Law_Enforcement_and_Protective_Services',
              'Marketing_and_Advertising',
              'Legal',
              'Weather',
              'Telecommunications',
              'Financial_Planning_and_Management',
              'Jobs_and_Career',
              'Universities_and_Colleges',
              'Water_Sports',
              'Horses',
              'Games',
              'Tourist_Attractions',
              'Astronomy',
              'Groceries',
              'Automotive_Industry',
              'Winter_Sports',
              'Investing',
              'Baseball',
              'History']


def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def visits(data):
    """
    non unique visits for last month
    """
    b = []
    err_counter = 0
    for i in range(data.shape[0]):
        try:
            timestamp_lst = list(map(lambda x: tm.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple()),
                                     list(data.iloc[i, 1].keys())))
            max_1 = max(timestamp_lst)
            mean = data.iloc[i, 1][datetime.fromtimestamp(
                max_1).strftime("%Y-%m-%d")]
            b.append([i, mean])
        except:
            b.append([i, -1])
            err_counter += 1
    print('visits: ', err_counter)
    return pd.Series(np.array(b).T[1])


def unique_users(data):
    """
    unique users visits for last month
    """
    err_counter = 0
    c = []
    for i in range(data.shape[0]):
        try:
            uu = data.loc[i, 'engagement'][list(data.loc[i, 'engagement'].keys())[
                0]]['UniqueUsers']
            if uu is None:
                c.append(-1)
            else:
                c.append(uu)
        except:
            err_counter += 1
            c.append(-1)
    print('unique_users: ', err_counter)
    return pd.Series(c)


def pages_per_visit(data):
    """
    pages per visit for last month
    """
    err_counter = 0
    d = []
    for i in range(data.shape[0]):
        try:
            uu = data.loc[i, 'engagement'][list(data.loc[i, 'engagement'].keys())[
                0]]['PagesPerVisit']
            if uu is None:
                d.append(-1)
            else:
                d.append(uu)
        except:
            err_counter += 1
            d.append(-1)
    print('pages_per_visit: ', err_counter)
    return pd.Series(d)


def bounce_rate(data):
    err_counter = 0
    e = []
    for i in range(data.shape[0]):
        try:
            uu = data.loc[i, 'engagement'][list(data.loc[i, 'engagement'].keys())[
                0]]['BounceRate']
            if uu is None:
                e.append(-1)
            else:
                e.append(uu)
        except:
            err_counter += 1
            e.append(-1)
    print('bounce_rate: ', err_counter)
    return pd.Series(e)


def avg_visit_duration(data):
    err_counter = 0
    f = []
    for i in range(data.shape[0]):
        try:
            uu = data.loc[i, 'engagement'][list(data.loc[i, 'engagement'].keys())[
                0]]['AvgVisitDuration']
            if uu is None:
                f.append(-1)
            else:
                f.append(uu)
        except:
            err_counter += 1
            f.append(-1)
    print('avg_visit_duration: ', err_counter)
    return pd.Series(f)


def alexa(data):
    h = []
    err_counter = 0
    for i in range(data.shape[0]):
        try:
            timestamp_lst = list(map(lambda x: tm.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple()),
                                     list(data.loc[i, 'alexa'].keys())))
            max_1 = max(timestamp_lst)
            mean = data.loc[i, 'alexa'][datetime.fromtimestamp(
                max_1).strftime("%Y-%m-%d")]
            h.append(mean)
        except:
            h.append(-1)
            err_counter += 1
    print('alexa: ', err_counter)
    return pd.Series(h)


def alexa_pop(data):
    h = []
    err_counter = 0
    for i in range(data.shape[0]):
        try:
            timestamp_lst = list(map(lambda x: tm.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple()),
                                     list(data.loc[i, 'alexa'].keys())))
            max_1 = max(timestamp_lst)
            mean = data.loc[i, 'alexa'][datetime.fromtimestamp(
                max_1).strftime("%Y-%m-%d")]
            h.append(mean['popularityText'])
        except:
            h.append(-1)
            err_counter += 1
    print('alexa_pop: ', err_counter)
    return pd.Series(h)


def category(data):
    h = []
    h1 = []

    err_counter = 0
    for i in range(data.shape[0]):
        h2 = []

        timestamp_lst = list(map(lambda x: tm.mktime(datetime.strptime(x, "%Y-%m-%d").timetuple()),
                                 list(data.loc[i, 'summary'].keys())))
        max_1 = max(timestamp_lst)
        mean = data.loc[i, 'summary'][datetime.fromtimestamp(
            max_1).strftime("%Y-%m-%d")]
        try:
            ind = list(mean['category']).index('/')
            h.append(''.join(list(mean['category'])[:ind]))
        except ValueError:
            ind = -1
            h.append(''.join(list(mean['category'])))
        try:
            h1.append(''.join(list(mean['category'])[ind + 1:]))
        except:
            h1.append(''.join(list(mean['category'])))

    print(err_counter)
    return pd.Series(h), pd.Series(h1)


def category_encoder(data):
    category1 = data[5]
    h = []
    try:
        ind = list(category1).index('/')
        h.append(''.join(list(category1)[:ind]))
        h.append(''.join(list(category1)[ind + 1:]))
    except:
        h.append(category1)

    cats = [0]*len(categories)
    category1 = h[0]
    cats[categories.index(category1)] = 1

    if len(h) == 2:
        category2 = h[1]
        cats[categories.index(category2)] = 1

    string = data[:5] + cats
    return string

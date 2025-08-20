import streamlit as st
import pandas as pd
import pickle
import numpy as np

# Muat daftar fitur yang telah disimpan
with open('selected_features.pkl', 'rb') as file:
    selected_features = pickle.load(file)

# Muat model yang telah dilatih
with open('supermarket.pkl', 'rb') as file:
    model = pickle.load(file)

# Judul aplikasi
st.set_page_config(page_title="PT.Supermarket Sejahtera", layout="centered", initial_sidebar_state="auto", page_icon="🏣")
st.title("""
Prediksi Profit Supermarket
\ndashboard was created by [Bramantio](https://www.linkedin.com/in/brahmantio-w/), here I want to try to introduce the results of my portfolio or my abilities in the field of data science. 
\nThis platform aims to provide an introduction, utilization, and exploration resources in the world of machine learning
""")

# Formulir untuk input data dari pengguna
st.subheader('Masukkan detail produk:')
Ship_mode = st.radio('Ship Mode', ['Standard Class', 'Second Class', 'First Class', 'Same Day'])
Segment = st.radio('Segment', ['Consumer','Corporate','Home Office'])
Country = st.radio('Country', ['United States'])
City = st.selectbox('City',['Henderson', 'Los Angeles', 'Fort Lauderdale', 'Concord',
        'Seattle', 'Fort Worth', 'Madison', 'West Jordan', 'San Francisco',
       'Fremont', 'Philadelphia', 'Orem', 'Houston', 'Richardson',
       'Naperville', 'Melbourne', 'Eagan', 'Westland', 'Dover',
       'New Albany', 'New York City', 'Troy', 'Chicago', 'Gilbert',
       'Springfield', 'Jackson', 'Memphis', 'Decatur', 'Durham',
       'Columbia', 'Rochester', 'Minneapolis', 'Portland', 'Saint Paul',
       'Aurora', 'Charlotte', 'Orland Park', 'Urbandale', 'Columbus',
       'Bristol', 'Wilmington', 'Bloomington', 'Phoenix', 'Roseville',
       'Independence', 'Pasadena', 'Newark', 'Franklin', 'Scottsdale',
       'San Jose', 'Edmond', 'Carlsbad', 'San Antonio', 'Monroe',
       'Fairfield', 'Grand Prairie', 'Redlands', 'Hamilton', 'Westfield',
       'Akron', 'Denver', 'Dallas', 'Whittier', 'Saginaw', 'Medina',
       'Dublin', 'Detroit', 'Tampa', 'Santa Clara', 'Lakeville',
       'San Diego', 'Brentwood', 'Chapel Hill', 'Morristown',
       'Cincinnati', 'Inglewood', 'Tamarac', 'Colorado Springs',
       'Belleville', 'Taylor', 'Lakewood', 'Arlington', 'Arvada',
       'Hackensack', 'Saint Petersburg', 'Long Beach', 'Hesperia',
       'Murfreesboro', 'Layton', 'Austin', 'Lowell', 'Manchester',
       'Harlingen', 'Tucson', 'Quincy', 'Pembroke Pines', 'Des Moines',
       'Peoria', 'Las Vegas', 'Warwick', 'Miami', 'Huntington Beach',
       'Richmond', 'Louisville', 'Lawrence', 'Canton', 'New Rochelle',
       'Gastonia', 'Jacksonville', 'Auburn', 'Norman', 'Park Ridge',
       'Amarillo', 'Lindenhurst', 'Huntsville', 'Fayetteville',
       'Costa Mesa', 'Parker', 'Atlanta', 'Gladstone', 'Great Falls',
       'Lakeland', 'Montgomery', 'Mesa', 'Green Bay', 'Anaheim',
       'Marysville', 'Salem', 'Laredo', 'Grove City', 'Dearborn',
       'Warner Robins', 'Vallejo', 'Mission Viejo', 'Rochester Hills',
       'Plainfield', 'Sierra Vista', 'Vancouver', 'Cleveland', 'Tyler',
       'Burlington', 'Waynesboro', 'Chester', 'Cary', 'Palm Coast',
       'Mount Vernon', 'Hialeah', 'Oceanside', 'Evanston', 'Trenton',
       'Cottage Grove', 'Bossier City', 'Lancaster', 'Asheville',
       'Lake Elsinore', 'Omaha', 'Edmonds', 'Santa Ana', 'Milwaukee',
       'Florence', 'Lorain', 'Linden', 'Salinas', 'New Brunswick',
       'Garland', 'Norwich', 'Alexandria', 'Toledo', 'Farmington',
       'Riverside', 'Torrance', 'Round Rock', 'Boca Raton',
       'Virginia Beach', 'Murrieta', 'Olympia', 'Washington',
       'Jefferson City', 'Saint Peters', 'Rockford', 'Brownsville',
       'Yonkers', 'Oakland', 'Clinton', 'Encinitas', 'Roswell',
       'Jonesboro', 'Antioch', 'Homestead', 'La Porte', 'Lansing',
       'Cuyahoga Falls', 'Reno', 'Harrisonburg', 'Escondido', 'Royal Oak',
       'Rockville', 'Coral Springs', 'Buffalo', 'Boynton Beach',
       'Gulfport', 'Fresno', 'Greenville', 'Macon', 'Cedar Rapids',
       'Providence', 'Pueblo', 'Deltona', 'Murray', 'Middletown',
       'Freeport', 'Pico Rivera', 'Provo', 'Pleasant Grove', 'Smyrna',
       'Parma', 'Mobile', 'New Bedford', 'Irving', 'Vineland', 'Glendale',
       'Niagara Falls', 'Thomasville', 'Westminster', 'Coppell', 'Pomona',
       'North Las Vegas', 'Allentown', 'Tempe', 'Laguna Niguel',
       'Bridgeton', 'Everett', 'Watertown', 'Appleton', 'Bellevue',
       'Allen', 'El Paso', 'Grapevine', 'Carrollton', 'Kent', 'Lafayette',
       'Tigard', 'Skokie', 'Plano', 'Suffolk', 'Indianapolis', 'Bayonne',
       'Greensboro', 'Baltimore', 'Kenosha', 'Olathe', 'Tulsa', 'Redmond',
       'Raleigh', 'Muskogee', 'Meriden', 'Bowling Green', 'South Bend',
       'Spokane', 'Keller', 'Port Orange', 'Medford', 'Charlottesville',
       'Missoula', 'Apopka', 'Reading', 'Broomfield', 'Paterson',
       'Oklahoma City', 'Chesapeake', 'Lubbock', 'Johnson City',
       'San Bernardino', 'Leominster', 'Bozeman', 'Perth Amboy',
       'Ontario', 'Rancho Cucamonga', 'Moorhead', 'Mesquite', 'Stockton',
       'Ormond Beach', 'Sunnyvale', 'York', 'College Station',
       'Saint Louis', 'Manteca', 'San Angelo', 'Salt Lake City',
       'Knoxville', 'Little Rock', 'Lincoln Park', 'Marion', 'Littleton',
       'Bangor', 'Southaven', 'New Castle', 'Midland', 'Sioux Falls',
       'Fort Collins', 'Clarksville', 'Sacramento', 'Thousand Oaks',
       'Malden', 'Holyoke', 'Albuquerque', 'Sparks', 'Coachella',
       'Elmhurst', 'Passaic', 'North Charleston', 'Newport News',
       'Jamestown', 'Mishawaka', 'La Quinta', 'Tallahassee', 'Nashville',
       'Bellingham', 'Woodstock', 'Haltom City', 'Wheeling',
       'Summerville', 'Hot Springs', 'Englewood', 'Las Cruces', 'Hoover',
       'Frisco', 'Vacaville', 'Waukesha', 'Bakersfield', 'Pompano Beach',
       'Corpus Christi', 'Redondo Beach', 'Orlando', 'Orange',
       'Lake Charles', 'Highland Park', 'Hempstead', 'Noblesville',
       'Apple Valley', 'Mount Pleasant', 'Sterling Heights', 'Eau Claire',
       'Pharr', 'Billings', 'Gresham', 'Chattanooga', 'Meridian',
       'Bolingbrook', 'Maple Grove', 'Woodland', 'Missouri City',
       'Pearland', 'San Mateo', 'Grand Rapids', 'Visalia',
       'Overland Park', 'Temecula', 'Yucaipa', 'Revere', 'Conroe',
       'Tinley Park', 'Dubuque', 'Dearborn Heights', 'Santa Fe',
       'Hickory', 'Carol Stream', 'Saint Cloud', 'North Miami',
       'Plantation', 'Port Saint Lucie', 'Rock Hill', 'Odessa',
       'West Allis', 'Chula Vista', 'Manhattan', 'Altoona', 'Thornton',
       'Champaign', 'Texarkana', 'Edinburg', 'Baytown', 'Greenwood',
       'Woonsocket', 'Superior', 'Bedford', 'Covington', 'Broken Arrow',
       'Miramar', 'Hollywood', 'Deer Park', 'Wichita', 'Mcallen',
       'Iowa City', 'Boise', 'Cranston', 'Port Arthur', 'Citrus Heights',
       'The Colony', 'Daytona Beach', 'Bullhead City', 'Portage', 'Fargo',
       'Elkhart', 'San Gabriel', 'Margate', 'Sandy Springs', 'Mentor',
       'Lawton', 'Hampton', 'Rome', 'La Crosse', 'Lewiston',
       'Hattiesburg', 'Danville', 'Logan', 'Waterbury', 'Athens',
       'Avondale', 'Marietta', 'Yuma', 'Wausau', 'Pasco', 'Oak Park',
       'Pensacola', 'League City', 'Gaithersburg', 'Lehi', 'Tuscaloosa',
       'Moreno Valley', 'Georgetown', 'Loveland', 'Chandler', 'Helena',
       'Kirkwood', 'Waco', 'Frankfort', 'Bethlehem', 'Grand Island',
       'Woodbury', 'Rogers', 'Clovis', 'Jupiter', 'Santa Barbara',
       'Cedar Hill', 'Norfolk', 'Draper', 'Ann Arbor', 'La Mesa',
       'Pocatello', 'Holland', 'Milford', 'Buffalo Grove', 'Lake Forest',
       'Redding', 'Chico', 'Utica', 'Conway', 'Cheyenne', 'Owensboro',
       'Caldwell', 'Kenner', 'Nashua', 'Bartlett', 'Redwood City',
       'Lebanon', 'Santa Maria', 'Des Plaines', 'Longview',
       'Hendersonville', 'Waterloo', 'Cambridge', 'Palatine', 'Beverly',
       'Eugene', 'Oxnard', 'Renton', 'Glenview', 'Delray Beach',
       'Commerce City', 'Texas City', 'Wilson', 'Rio Rancho', 'Goldsboro',
       'Montebello', 'El Cajon', 'Beaumont', 'West Palm Beach', 'Abilene',
       'Normal', 'Saint Charles', 'Camarillo', 'Hillsboro', 'Burbank',
       'Modesto', 'Garden City', 'Atlantic City', 'Longmont', 'Davis',
       'Morgan Hill', 'Clifton', 'Sheboygan', 'East Point', 'Rapid City',
       'Andover', 'Kissimmee', 'Shelton', 'Danbury', 'Sanford',
       'San Marcos', 'Greeley', 'Mansfield', 'Elyria', 'Twin Falls',
       'Coral Gables', 'Romeoville', 'Marlborough', 'Laurel', 'Bryan',
       'Pine Bluff', 'Aberdeen', 'Hagerstown', 'East Orange',
       'Arlington Heights', 'Oswego', 'Coon Rapids', 'San Clemente',
       'San Luis Obispo', 'Springdale', 'Lodi', 'Mason'])
State = st.selectbox('State',['Kentucky', 'California', 'Florida', 'North Carolina',
                'Washington', 'Texas', 'Wisconsin', 'Utah', 'Nebraska',
                'Pennsylvania', 'Illinois', 'Minnesota', 'Michigan', 'Delaware',
                'Indiana', 'New York', 'Arizona', 'Virginia', 'Tennessee',
                'Alabama', 'South Carolina', 'Oregon', 'Colorado', 'Iowa', 'Ohio',
                'Missouri', 'Oklahoma', 'New Mexico', 'Louisiana', 'Connecticut',
                'New Jersey', 'Massachusetts', 'Georgia', 'Nevada', 'Rhode Island',
                'Mississippi', 'Arkansas', 'Montana', 'New Hampshire', 'Maryland',
                'District of Columbia', 'Kansas', 'Vermont', 'Maine',
                'South Dakota', 'Idaho', 'North Dakota', 'Wyoming',
                'West Virginia'])
Postal_code = st.number_input('Postal_code',
                           min_value=0,
                           step=1,)
Region = st.radio('Region',['South', 'West', 'Central', 'East'])
Category = st.radio('Category',['Furniture', 'Office Supplies', 'Technology'])
Sub_Category = st.selectbox('Sub_Category',['Bookcases', 'Chairs', 'Labels', 'Tables', 'Storage',
                            'Furnishings', 'Art', 'Phones', 'Binders', 'Appliances', 'Paper',
                            'Accessories', 'Envelopes', 'Fasteners', 'Supplies', 'Machines',
                            'Copiers'])
Sales = st.number_input('Sales', value=100.0)
Quantity = st.number_input('Quantity', value=1, min_value=1)
Discount = st.number_input('Discount', value=0.0, min_value=0.0)

if st.button('Prediksi Profit'):
    # Kumpulkan data input
    input_data = {
        'Ship_mode': Ship_mode,
        'Segment': Segment,
        'Country': Country,
        'City':City,
        'State':State,
        'Postal_code':Postal_code,
        'Region':Region,
        'Category':Category,
        'Sub_Category':Sub_Category,
        'Sales': Sales,
        'Quantity': Quantity,
        'Discount': Discount,
    }
    
    features = pd.DataFrame([input_data])
    
    # Lakukan semua langkah preprocessing yang sama seperti saat pelatihan
    #feature combination
    features['revenue_item'] = features['Sales'] / features['Quantity']

    #log transform
    features['Sales'] = np.log1p(features['Sales'])
    features['Postal_code'] = np.log1p(features['Postal_code'])
    features['revenue_item'] = np.log1p(features['revenue_item'])

    # encoding
    le = LabelEncoder()
    features['Ship_mode_emb'] = le.fit_transform(features['Ship_mode'])
    features['Segment_emb'] = le.fit_transform(features['Segment'])
    features['Country_emb'] = le.fit_transform(features['Country'])
    features['City_emb'] = le.fit_transform(features['City'])
    features['State_emb'] = le.fit_transform(features['State'])
    features['Region_emb'] = le.fit_transform(features['Region'])
    features['Category_emb'] = le.fit_transform(features['Category'])
    features['Sub_Category_emb'] = le.fit_transform(features['Sub_Category'])

    # Pilih fitur-fitur yang sudah diseleksi
    features = features[selected_features]
    
    # Lakukan prediksi
    prediction_log = model.predict(features)
    prediction = np.expm1(prediction_log)[0]
    
    # Tampilkan hasil
    st.subheader('Hasil Prediksi:')
    if prediction > 0:
        st.success(f"Profit yang diprediksi: **${prediction:,.2f}** (Untung)")
    else:
        st.error(f"Profit yang diprediksi: **${prediction:,.2f}** (Rugi)")

import streamlit as st
import pickle
import pandas as pd

# Define teams and cities lists
teams = ['Sunrisers Hyderabad', 'Mumbai Indians', 'Royal Challengers Bangalore',
         'Kolkata Knight Riders', 'Kings XI Punjab', 'Chennai Super Kings',
         'Rajasthan Royals', 'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

# Load the pickled model
with open('ipltwo.pkl', 'rb') as f:
    pipe = pickle.load(f)

# Streamlit app title
st.title('Kon Jitega IPL')

# Add some custom CSS for layout
st.markdown(
    """
    
    """
)

# Create a container for layout
st.write('<div>', unsafe_allow_html=True)

# Select batting team
batting_team = st.selectbox('Select the batting team', teams, key='batting_team')

# Select bowling team
bowling_team = st.selectbox('Select the bowling team', teams, key='bowling_team')

# Select host city
selected_city = st.selectbox('Select host city', cities)

# Input target, score, overs completed, and wickets
target = st.number_input('Target')
score = st.number_input('Score')
overs = st.number_input('Overs completed')
wickets = st.number_input('Wickets out')

# Close the container
st.write('</div>', unsafe_allow_html=True)

# Button to trigger the prediction
if st.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs * 6)
    wickets = 10 - wickets
    crr = score / overs
    rrr = (runs_left * 6) / balls_left

    input_data = {'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city],
                  'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets],
                  'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]}
    input_df = pd.DataFrame(input_data)

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]

    st.header(batting_team + "- " + str(round(win * 100)) + "%")
    st.header(bowling_team + "- " + str(round(loss * 100)) + "%")
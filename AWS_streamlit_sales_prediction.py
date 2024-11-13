import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from scipy.stats import ttest_ind
import statsmodels.api as sm
import datetime
import base64

mse = MeanSquaredError()

# Load the model with custom objects explicitly defined
model_for_weekly_sales = tf.keras.models.load_model('LSTM_model_for_sales_without_MD.h5',custom_objects={'mse': mse})
scaler_for_weekly_sales= joblib.load('weekly_sales_scaler.pkl')

st.set_page_config(layout='wide')




def set_bg_hack(main_bg):
    '''
    A function to unpack an image from root folder and set as bg.
    Returns
    -------
    The background.
    '''
    # set bg name
    main_bg_ext = "jpg"

    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url(data:image/{main_bg_ext};base64,{main_bg});
             background-size: cover
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

# Load your image
with open(r"D:\GUVI_projects\06_Retail Sales\OIP.jpeg", "rb") as image_file:
    image_bytes = image_file.read()
    encoded_image = base64.b64encode(image_bytes).decode()

# Set the background image
set_bg_hack(encoded_image)

st.title(":red[Retail Sales Price Prediction]")
tabs=st.tabs(['Input','Prediction','Sales trend','Inventory Management','Insights','Markdown Impact'])

with tabs[0]:
    st.write('Kindly upload the file with the sales value in the columns name "Weekly_Sales" and Date in the format "YYYY-MM-DD". And also check if there is any null values, If Yes, then Kindly fill it and upload the file.')
    st.write('We just need the date and Weekly sales for the past twelve weeks as the lookback value is taken as 12 for predicting the sales till your expected week')
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xls", "xlsx"])
    if uploaded_file is not None:
    # Load the file into a DataFrame based on file type
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        st.write("Here is a preview of your data:")
        st.write(data.head())
        if data['Weekly_Sales'].isna().any():
            st.warning("Warning: Missing values found in 'Weekly_Sales' column. Filling with last valid value.")

    else:
        st.warning("Please upload a file to proceed.")

with tabs[1]:
    No_of_weeks=st.number_input('Enter the number of weeks for prediction:', min_value= 1)
    lookback=12
    store_number = st.number_input('Enter the store number:', min_value=1, max_value=45)
    deprt_number = st.number_input('Enter the depart number:', min_value=1, max_value = 99)
    data.loc[data['Weekly_Sales'] < 0, 'Weekly_Sales'] = 0
    df=data[(data['Store']==store_number)&(data['Dept']== deprt_number)]
    df['Date'] = pd.to_datetime(df['Date'])
    col1, col2=st.columns(2)
    if len(df)>12:
        current_input =df[['Date','Weekly_Sales']]
        predict_for=No_of_weeks
        col1.write(current_input)
        tabs[2].header('Trend of Input Data')
        tabs[2].line_chart(current_input, x='Date', y='Weekly_Sales')
        current_input_1 =scaler_for_weekly_sales.transform(df['Weekly_Sales'].values.reshape(-1,1))
        last_date = df['Date'].iloc[-1]
        prediction_dates = []
        for i in range (predict_for):
            this_input=current_input_1[-lookback:]
            this_input=this_input.reshape((1,1,12))
            print(this_input)
            print(this_input.shape)
            this_prediction=model_for_weekly_sales.predict(this_input)
            print(this_prediction)
            current_input_1=np.append(current_input_1,this_prediction.flatten())
            last_date += pd.Timedelta(days=7)
            prediction_dates.append(last_date)
        predict_on_future=np.reshape(np.array(current_input_1[-predict_for:]),(predict_for,1))
        predict_on_future=pd.DataFrame(scaler_for_weekly_sales.inverse_transform(predict_on_future),columns= ['Weekly_Sales'])
        predict_on_future['Date'] = prediction_dates
        col2.write(predict_on_future[['Date','Weekly_Sales']])
        tabs[2].header('Trend of Predicted Data')
        tabs[2].line_chart(predict_on_future, x='Date', y='Weekly_Sales',color='#FF0000')
    else:
        st.warning('Inadequate values in selected departments and store')
    
with tabs[3]:
    predict_on_future['Date']=pd.to_datetime(predict_on_future['Date'])
    predict_on_future['Date']=predict_on_future['Date'].dt.date
    date = st.date_input('Enter the date of stock to be predicted:', format='YYYY-MM-DD', min_value=datetime.date(2010, 4, 30))
    if date in predict_on_future['Date'].values:
        df_p=predict_on_future[predict_on_future['Date']==date]
        stock = st.number_input('Enter the stock on the expected date', min_value=1)
        predicted_sales = round((df_p['Weekly_Sales'].values[0]))
        if stock<predicted_sales:
            purchase = float(predicted_sales) - float(stock)
            st.write(f'{purchase} amount of stock to be purchased to meet the predicted sale of {predicted_sales}')
        elif stock == predicted_sales:
            st.write(f'stock in hand is equal to the predicted sale value{predicted_sales}, better to purchase more')
        else:
            st.write(f' you already have the stock to meet the predicted sales of {predicted_sales}')        
    else:
        st.warning('The date is not within the predicted week')
with tabs[4]:
    st.write('Here we are going to see the insights we can learn in the process of Predicting')
    higher_sales = data.sort_values(by='Weekly_Sales', ascending=False)
    lower_sales = data.sort_values(by='Weekly_Sales', ascending = True)
    st.write('highest weekly sales in the given data:',higher_sales.head(1))

    st.write('Holiday is the prime factor which impacts sales of the week, The highest sales is',higher_sales[['Weekly_Sales','IsHoliday_x']].head(1))
    st.write('and the lowest sales is',lower_sales[['Weekly_Sales', 'IsHoliday_x']].iloc[1].to_frame().T)
    st.write('In IsHoliday_x, 1 represents True and 0 represents False')
    
    st.write('The impact of Markdowns is also have some effect on Weekly Sales')
with tabs[5]:
    MD_i_tabs=st.tabs(['Visually','Statistically','Impact insights'])
    with MD_i_tabs[0]:
        col3, col4 =st.columns(2)
        col3.write("Summary statistics for non-holiday weeks:")
        col3.write(data[data['IsHoliday_x'] == 0]['Weekly_Sales'].describe())

        col4.write("\nSummary statistics for holiday weeks:")
        col4.write(data[data['IsHoliday_x'] == 1]['Weekly_Sales'].describe())

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='IsHoliday_x', y='Weekly_Sales', data=data, ax=ax)
        ax.set_title('Weekly Sales Distribution for Holiday vs Non-Holiday Weeks')
        ax.set_xlabel('Is Holiday (0=Non-Holiday, 1=Holiday)')
        ax.set_ylabel('Weekly Sales')
        st.pyplot(fig)
    with MD_i_tabs[1]:
        MD_i_tabs_stats=st.tabs(['T-Test','Linearity test'])
        with MD_i_tabs_stats[0]:
            data['mean_markdown'] = data[['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']].mean(axis=1)
            holiday_sales = data[(data['IsHoliday_x'] == 1) & (data['mean_markdown'] > 0)]['Weekly_Sales']
            non_holiday_sales = data[(data['IsHoliday_x'] == 0) & (data['mean_markdown'] > 0)]['Weekly_Sales']
            t_stat, p_value = ttest_ind(holiday_sales, non_holiday_sales, equal_var=False)

            st.header(f"T-statistic: {t_stat}")
            st.header(f"P-value: {p_value}")

            if p_value < 0.05:
                st.header("Reject the null hypothesis: There is a statistically significant difference in sales due to markdowns on holiday vs. non-holiday weeks.")
            else:
                st.header("Fail to reject the null hypothesis: No significant difference in sales due to markdowns on holiday vs. non-holiday weeks.")
            st.header('The T-statistic should be nearer to 0 and P_value should be less than significance level to Fail to reject the null hypothesis and so null hypothesis is rejected')      
        with MD_i_tabs_stats[1]:
            # Create an interaction term between mean_markdown and is_holiday
            data['interaction'] = data['mean_markdown'] * data['IsHoliday_x']

            # Prepare the regression data
            X = data[['mean_markdown', 'IsHoliday_x', 'interaction']]
            y = data['Weekly_Sales']

            # Add a constant to the independent variables
            X = sm.add_constant(X)

            # Fit the regression model
            model_lin_reg= sm.OLS(y, X).fit()
            col5, col6 = st.columns(2)
            # Display the summary of regression results
            col5.write(model_lin_reg.summary())
            col6.write('''
                    R-squared and Adjusted R-squared:
                        
                    Both are near zero, indicating the model explains almost none of the variance in Weekly_Sales. This suggests that markdowns, holiday indicators, and their interaction are not strong predictors of weekly sales on their own, or other unaccounted variables may drive sales.
                    
                    Coefficients:

                    Intercept (const): 15,890. This represents the baseline average weekly sales when all predictors are zero, which is a reference level and doesn’t provide insight on markdown or holiday effects.
                    
                    mean_markdown: Coefficient is -0.0026 with a p-value of 0.239, which is not statistically significant. This suggests that markdown size alone does not significantly affect weekly sales.
                    
                    IsHoliday_x: Coefficient of 1080.60 with a p-value of 0.000. This shows that during holiday weeks, sales are estimated to increase by approximately 1080 units compared to non-holiday weeks. This effect is statistically significant.
                    
                    Interaction (mean markdown * holiday): Coefficient of 0.0253 with a p-value of 0.013. This suggests a small but significant positive effect of markdowns specifically during holiday weeks. For every unit increase in markdown during a holiday, sales increase slightly.''')
    with MD_i_tabs[2]:
        st.write('Holiday Effect:') 
        st.write('Holiday weeks alone have a positive, significant impact on sales.')
        st.write('Markdown Effect:') 
        st.write("The direct effect of mean_markdown is not significant.")
        st.write('Interaction Effect:') 
        st.write('Markdown effectiveness is significantly higher during holiday weeks, suggesting markdowns may work best as a sales-boosting strategy when timed with holidays.')
        
        st.header('This model indicates that while markdowns alone don’t significantly drive sales, they are more impactful when used during holiday weeks, which is valuable for strategic planning around sales and promotions.')


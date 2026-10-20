import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


drivers = [
    'Monthly_Rainfall_mm', 
    'Temp_Anomaly_C', 
    'Regional_Fuel_Price_Rs_Ltr', 
    'Transport_Cost_Rs_Qtl', 
    'Market_Arrivals_Tonnes', 
    'Cost_of_Production_Rs_Qtl'
]

targets = ['Min_Wholesale_Price_Rs_Qtl', 'Price_Volatility_Index_Pct']

def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    print(df.isna().sum())
    df = df.sort_values(by=['APMC_Mandi', 'Commodity', 'Date']).reset_index(drop=True)
    return df

df = load_and_preprocess_data('APMC_Macroeconomic_Environmental_Price_Volatility_Dataset.csv')

def generate_summary_tables(df):

    table1 = pd.DataFrame({
        'Feature Name': df.columns,
        'Data Type': df.dtypes.values,
        'Missing Values': df.isnull().sum().values,
        'Missing %': (df.isnull().sum() / len(df) * 100).values,
        # FIXED: Replacing np.issubdtype with pd.api.types.is_numeric_dtype
        'Min': [df[col].min() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A' for col in df.columns],
        'Max': [df[col].max() if pd.api.types.is_numeric_dtype(df[col]) else 'N/A' for col in df.columns]
    })
   
    outlier_list = []
    for comm in df['Commodity'].unique():
        sub = df[df['Commodity'] == comm]['Min_Wholesale_Price_Rs_Qtl']
        q1, q3 = sub.quantile(0.25), sub.quantile(0.75)
        iqr = q3 - q1
        lower_bound, upper_bound = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers_iqr = ((sub < lower_bound) | (sub > upper_bound)).sum()
        z_scores = stats.zscore(sub)
        outliers_z = (np.abs(z_scores) > 3).sum()
        outlier_list.append({
            'Commodity': comm,
            'Q1': q1, 'Q3': q3, 'IQR': iqr,
            'Lower Bound': lower_bound, 'Upper Bound': upper_bound,
            'IQR Outliers': outliers_iqr, 'Z-Score Outliers (|Z|>3)': outliers_z
        })
    table2 = pd.DataFrame(outlier_list)
   
    table3 = df.groupby(['State', 'Commodity']).agg(
        Avg_Arrivals_Tonnes=('Market_Arrivals_Tonnes', 'mean'),
        Mean_Min_Price=('Min_Wholesale_Price_Rs_Qtl', 'mean'),
        Mean_Volatility_Pct=('Price_Volatility_Index_Pct', 'mean'),
        Volatility_Std=('Price_Volatility_Index_Pct', 'std')
    ).reset_index()
    
    corr_results = []
    for target in targets:
        for driver in drivers:
            p_corr, p_val = stats.pearsonr(df[driver], df[target])
            s_corr, s_val = stats.spearmanr(df[driver], df[target])
            corr_results.append({
                'Target Variable': target,
                'Driver Variable': driver,
                'Pearson r': round(p_corr, 4),
                'Pearson p-value': round(p_val, 5),
                'Spearman rho': round(s_corr, 4),
                'Spearman p-value': round(s_val, 5),
                'Significant (p < 0.05)': p_val < 0.05
            })
    table4 = pd.DataFrame(corr_results)
    
    
generate_summary_tables(df)


fig, axes = plt.subplots(
    1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3.5, 1]}
)

df_regular = df[df['Min_Wholesale_Price_Rs_Qtl'] < 20000]
sns.boxplot(
    data=df_regular,
    x='Commodity',
    y='Min_Wholesale_Price_Rs_Qtl',
    hue='State',
    ax=axes[0],
)
axes[0].set_title(
    'Standard APMC Commodities (< ₹20,000/Qtl)', fontsize=12, fontweight='bold'
)
axes[0].tick_params(axis='x', rotation=45)
axes[0].set_ylabel('Min Wholesale Price (Rs/Qtl)', fontsize=11)
axes[0].legend(title='State', loc='upper left', fontsize=9)


df_high = df[df['Min_Wholesale_Price_Rs_Qtl'] >= 20000]
sns.boxplot(
    data=df_high,
    x='Commodity',
    y='Min_Wholesale_Price_Rs_Qtl',
    hue='State',
    ax=axes[1],
    palette=['#7FA1C3'],
)
axes[1].set_title(
    'High-Value Crop (Arecanut)', fontsize=12, fontweight='bold'
)
axes[1].set_ylabel('')
axes[1].legend(title='State', loc='upper left', fontsize=9)

plt.suptitle(
    'Visual 1: Minimum Wholesale Price Distribution across APMC Commodities',
    fontsize=14,
    fontweight='bold',
)
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = ['Min_Wholesale_Price_Rs_Qtl', 'Price_Volatility_Index_Pct', 'Market_Arrivals_Tonnes', 'Temp_Anomaly_C']
titles = ['Min Wholesale Price', 'Price Volatility Index (%)', 'Market Arrivals (Tonnes)', 'Temperature Anomaly (°C)']

for ax, feature, title in zip(axes.flatten(), features, titles):
    sns.histplot(df[feature], kde=True, ax=ax, color='teal', bins=30)
    ax.set_title(f'Distribution: {title}', fontsize=12, fontweight='bold')
    ax.set_xlabel(title)
    ax.set_ylabel('Frequency')

plt.suptitle('Visual 2: Feature Normality Diagnostics', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()


plt.figure(figsize=(14, 6))
top_commodities = ['Onion', 'Tomato', 'Wheat', 'Paddy (Dhan)']
df_top = df[df['Commodity'].isin(top_commodities)]
sns.lineplot(data=df_top, x='Date', y='Min_Wholesale_Price_Rs_Qtl', hue='Commodity', errorbar=None, linewidth=2)
plt.title('Visual 3: Multi-Year Price Trajectories (2018–2025)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Min Wholesale Price (Rs/Qtl)', fontsize=12)
plt.tight_layout()
plt.show()


plt.figure(figsize=(10, 6))
pivoted_vol = df.pivot_table(index='State', columns='Year', values='Price_Volatility_Index_Pct', aggfunc='mean')
sns.heatmap(pivoted_vol, annot=True, fmt=".2f", cmap='YlOrRd', cbar_kws={'label': 'Volatility Index (%)'})
plt.title('Visual 4: State-Wise Annual Price Volatility Heatmap', fontsize=14, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('State', fontsize=12)
plt.tight_layout()
plt.show()


monthly_agg = df.groupby('Month')[['Market_Arrivals_Tonnes', 'Min_Wholesale_Price_Rs_Qtl']].mean()
months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                'July', 'August', 'September', 'October', 'November', 'December']
monthly_agg = monthly_agg.reindex(months_order)

fig, ax1 = plt.subplots(figsize=(12, 6))
color = 'tab:blue'
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Avg Market Arrivals (Tonnes)', color=color, fontsize=12)
ax1.bar(monthly_agg.index, monthly_agg['Market_Arrivals_Tonnes'], color=color, alpha=0.4, label='Arrivals')
ax1.tick_params(axis='y', labelcolor=color)
plt.xticks(rotation=45)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('Avg Min Wholesale Price (Rs/Qtl)', color=color, fontsize=12)
ax2.plot(monthly_agg.index, monthly_agg['Min_Wholesale_Price_Rs_Qtl'], color=color, marker='o', linewidth=2.5, label='Min Price')
ax2.tick_params(axis='y', labelcolor=color)

plt.title('Visual 5: Seasonal Inverse Relationship (Arrivals vs. Price)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()


plt.figure(figsize=(11, 8))
num_cols = ['Min_Wholesale_Price_Rs_Qtl', 'Modal_Wholesale_Price_Rs_Qtl', 'Max_Wholesale_Price_Rs_Qtl',
            'Price_Volatility_Index_Pct', 'Market_Arrivals_Tonnes', 'Monthly_Rainfall_mm', 
            'Temp_Anomaly_C', 'Regional_Fuel_Price_Rs_Ltr', 'Transport_Cost_Rs_Qtl', 'Cost_of_Production_Rs_Qtl']
corr_matrix = df[num_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Visual 6: Multi-Factor Pearson Correlation Heatmap Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='Transport_Cost_Rs_Qtl', y='Min_Wholesale_Price_Rs_Qtl',
            scatter_kws={'alpha':0.3, 'color':'gray'}, line_kws={'color':'red', 'linewidth':2})
plt.title('Visual 7: Transport Cost Pass-Through Elasticity Regression Plot', fontsize=14, fontweight='bold')
plt.xlabel('Transport Cost (Rs/Qtl)', fontsize=12)
plt.ylabel('Min Wholesale Price (Rs/Qtl)', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Temp_Anomaly_C', y='Price_Volatility_Index_Pct', hue='Commodity', alpha=0.7)
sns.lineplot(data=df, x='Temp_Anomaly_C', y='Price_Volatility_Index_Pct', color='black', linewidth=2, ci=None)
plt.title('Visual 8: Climate Shock vs. Price Volatility Index', fontsize=14, fontweight='bold')
plt.xlabel('Temperature Anomaly (°C)', fontsize=12)
plt.ylabel('Price Volatility Index (%)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


ts_data = df[(df['APMC_Mandi'] == 'Lasalgaon APMC') & (df['Commodity'] == 'Onion')].copy()
ts_data = ts_data.set_index('Date').asfreq('MS')

# Visual 9: Time-Series Seasonal Decomposition
decomp = seasonal_decompose(ts_data['Min_Wholesale_Price_Rs_Qtl'], model='additive', period=12)
fig = decomp.plot()
fig.set_size_inches(12, 8)
fig.suptitle('Visual 9: Time-Series Seasonal Decomposition (Lasalgaon Onion)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.show()

# Visual 10: Autocorrelation (ACF) & Partial Autocorrelation (PACF) Plots
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
plot_acf(ts_data['Min_Wholesale_Price_Rs_Qtl'], lags=24, ax=axes[0])
plot_pacf(ts_data['Min_Wholesale_Price_Rs_Qtl'], lags=24, ax=axes[1])
axes[0].set_title('Autocorrelation (ACF)', fontweight='bold')
axes[1].set_title('Partial Autocorrelation (PACF)', fontweight='bold')
plt.suptitle('Visual 10: ACF & PACF Lag Identification', fontsize=14, fontweight='bold', y=1.05)
plt.tight_layout()
plt.show()

# Train / Test Split (Train: 2018-2024, Test: 2025)
train = ts_data[ts_data.index < '2025-01-01']
test = ts_data[ts_data.index >= '2025-01-01']

# Fit SARIMAX Model
exog_cols = ['Transport_Cost_Rs_Qtl', 'Market_Arrivals_Tonnes', 'Temp_Anomaly_C']
sarimax_model = SARIMAX(train['Min_Wholesale_Price_Rs_Qtl'], 
                        exog=train[exog_cols], 
                        order=(1, 1, 1), 
                        seasonal_order=(1, 1, 0, 12)).fit(disp=False)

forecast_res = sarimax_model.get_forecast(steps=len(test), exog=test[exog_cols])
forecast_mean = forecast_res.predicted_mean
conf_int = forecast_res.conf_int()

plt.figure(figsize=(12, 6))
plt.plot(train.index, train['Min_Wholesale_Price_Rs_Qtl'], label='Train Historical', color='blue')
plt.plot(test.index, test['Min_Wholesale_Price_Rs_Qtl'], label='Test Actual Ground Truth', color='black', linewidth=2)
plt.plot(test.index, forecast_mean, label='SARIMAX Forecast', color='red', linestyle='--')
plt.fill_between(test.index, conf_int.iloc[:, 0], conf_int.iloc[:, 1], color='pink', alpha=0.5, label='95% Confidence Interval')
plt.title('Visual 11: SARIMAX Model Out-of-Sample Price Forecast (Lasalgaon Onion)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Min Wholesale Price (Rs/Qtl)', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

# Fit Random Forest Regressor & Visual 12: Feature Importance
X_train, y_train = train[drivers], train['Min_Wholesale_Price_Rs_Qtl']
X_test, y_test = test[drivers], test['Min_Wholesale_Price_Rs_Qtl']

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

importances = pd.Series(rf_model.feature_importances_, index=drivers).sort_values(ascending=True)

# Visual 12: Driver Weight / Feature Importance Bar Chart
plt.figure(figsize=(10, 5))
importances.plot(kind='barh', color='darkgreen')
plt.title('Visual 12: Random Forest Multi-Factor Feature Importance', fontsize=14, fontweight='bold')
plt.xlabel('Relative Importance Score', fontsize=12)
plt.ylabel('Driver Variable', fontsize=12)
plt.tight_layout()
plt.show()

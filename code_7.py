import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

#If any of this libraries is missing from your computer. Please install them using pip.

filename = 'Flight_Delays_2018.csv'
df = pd.read_csv("Flight_Delays_2018.csv")

print(df.head())

df = df[['ARR_DELAY', 'DEP_DELAY','DISTANCE','AIR_TIME']]
df = df.dropna()

print(df.describe)

# create a scatter plot 
plt.scatter(df['DEP_DELAY'], df['ARR_DELAY'])
plt.xlabel("Departure Delay")
plt.ylabel("Arrival Delay")
plt.title("Departure vs Arrival Delay")
plt.show()

plt.scatter(df['AIR_TIME'], df['ARR_DELAY'])
plt.xlabel("Air Time")
plt.ylabel("Arrival Delay") 
plt.title("Air Time vs Arrival Delay")
plt.show()

#predictive analytics 
X = df[['DEP_DELAY', 'DISTANCE', 'AIR_TIME']]

X = sm.add_constant(X)

Y = df['ARR_DELAY']
model_Simple = sm.OLS(Y, X).fit()

print("OLS Summary")
print(model_Simple.summary())

fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(model_Simple, 1, ax=ax)
ax.set_ylabel("Arrival Delay")
ax.set_title(" OLS Visualization: ")
plt.show()


#ARR_DELAY is the column name that should be used as dependent variable (Y).
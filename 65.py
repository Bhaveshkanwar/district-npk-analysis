
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("/Users/bhaveshkanwar/Desktop/ICRISAT-District Level Data.csv")
df.columns = df.columns.str.strip()  


fertilizers = [
    'NITROGEN PER HA OF GCA (Kg per ha)',
    'PHOSPHATE PER HA OF GCA (Kg per ha)',
    'POTASH PER HA OF GCA (Kg per ha)'
]


df.dropna(subset=fertilizers, inplace=True)
df['Total NPK (kg/ha)'] = df[fertilizers].sum(axis=1)


print(df.info())
print(df.head())


print(df[fertilizers].describe())


print(df.isnull().sum())


for fert in fertilizers:
    plt.figure(figsize=(8, 4))
    sns.histplot(df[fert], kde=True, bins=30)
    plt.title(f'Distribution of {fert}')
    plt.xlabel(fert)
    plt.grid(True)
    plt.show()


corr_matrix = df[fertilizers].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='YlGnBu', fmt='.2f',
            xticklabels=fertilizers, yticklabels=fertilizers)
plt.title('Correlation Heatmap of NPK Usage (kg/ha)', fontsize=14)
plt.xticks(rotation=45, ha='right')  
plt.yticks(rotation=0)              
plt.tight_layout()                   
plt.show()



for fert in fertilizers:
    plt.figure(figsize=(6, 1.5))
    sns.boxplot(x=df[fert])
    plt.title(f'Outliers in {fert}')
    plt.grid(True)
    plt.show()


plt.figure(figsize=(12, 6))
sns.violinplot(x='State Name', y='NITROGEN PER HA OF GCA (Kg per ha)', data=df, inner='quart')
plt.title("Violin Plot of Nitrogen Usage by State")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12,6))
top_states = df['State Name'].value_counts().index[:5]
for state in top_states:
    yearly = df[df['State Name'] == state].groupby('Year')['NITROGEN PER HA OF GCA (Kg per ha)'].mean()
    plt.plot(yearly.index, yearly.values, label=state)

plt.title('Nitrogen Usage Over Years by Top 5 States')
plt.xlabel('Year')
plt.ylabel('Nitrogen (kg/ha)')
plt.legend()
plt.grid(True)
plt.show()


plt.figure(figsize=(8, 5))
sns.regplot(x='TOTAL PER HA OF GCA (Kg per ha)', y='TOTAL CONSUMPTION (tons)', data=df,
            scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title("Regression: Total Usage per ha vs Total Consumption (tons)")
plt.grid(True)
plt.show()




df['Total NPK (kg/ha)'] = df[fertilizers].sum(axis=1)
top_districts = df[['Dist Name', 'State Name', 'Total NPK (kg/ha)']].sort_values(by='Total NPK (kg/ha)', ascending=False).head(10)

top_districts['dummy'] = 'NPK' 
sns.barplot(x='Total NPK (kg/ha)', y='Dist Name', data=top_districts, hue='dummy', palette='magma', dodge=False)
plt.legend().remove()
plt.title('Top 10 Districts by Total NPK Usage (kg/ha)')
plt.xlabel('Total NPK (kg/ha)')
plt.ylabel('District')
plt.grid(True)
plt.tight_layout()
plt.show()



stacked_top = df[['Dist Name'] + fertilizers].sort_values(by=fertilizers[0], ascending=False).head(10).set_index('Dist Name')
stacked_top.plot(kind='bar', stacked=True, figsize=(12, 6), colormap='coolwarm')
plt.title("Stacked Bar Chart of NPK Usage for Top Districts")
plt.ylabel("Fertilizer Use (kg/ha)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


sns.pairplot(df[fertilizers])
plt.suptitle("Pair Plot of NPK Usage", y=1.02)
plt.show()


avg_vals = df[fertilizers].mean()
plt.figure(figsize=(6,6))
plt.pie(avg_vals, labels=['Nitrogen', 'Phosphate', 'Potash'], autopct='%1.1f%%', startangle=140, explode=[0.05]*3)
plt.title("Average Share of NPK Usage")
plt.axis('equal')
plt.show()


heatmap_data = df.pivot_table(index='State Name', columns='Year', values='NITROGEN PER HA OF GCA (Kg per ha)', aggfunc='mean')
plt.figure(figsize=(14, 8))
sns.heatmap(heatmap_data, cmap='YlOrRd', linewidths=0.5)
plt.title('Nitrogen Usage Heatmap (kg/ha) by State and Year')
plt.show()

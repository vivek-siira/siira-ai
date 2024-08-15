import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd

# Sample data
data = {
    'daypart': ['Late Fringe', 'Early Fringe', 'Early Morning', 'Prime', 'Weekend', 'Overnight', 'Daytime'],
    'delivery': [348125.0, 737146.0, 159546.0, 1348543.0, 477739.0, 277772.0, 1018467.0]
}
csv_data = pd.DataFrame(data)

# Sort the data in descending order by delivery
csv_data = csv_data.sort_values('delivery', ascending=False)

# Adjust figure size based on the length of x-axis labels
label_length = max(len(label) for label in csv_data['daypart'])
fig_width = 8 + 0.2 * label_length  # Adjust width dynamically
fig, ax = plt.subplots(figsize=(fig_width, 6))

# Creating a bar plot
ax.bar(csv_data['daypart'], csv_data['delivery'], color='blue')

# Setting the title and labels
ax.set_title('Delivery by Daypart')
ax.set_xlabel('Daypart')
ax.set_ylabel('Delivery')

# Setting the yticks to display approximate values
ax.set_yticks([0, 200000, 400000, 600000, 800000, 1000000, 1200000, 1400000])
ax.set_yticklabels(['0', '200K', '400K', '600K', '800K', '1M', '1.2M', '1.4M'])

# Rotate the x-axis labels for better readability
plt.xticks(rotation=45, ha='right')

# Display the plot using Streamlit
st.pyplot(fig)

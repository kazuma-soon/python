import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import basic_plan
import percent_plan
import down_plan

# price.size -> 121, basic_plan_invest.size -> 120
# price = utils.gbm()[:-1]
basic_plan_invest,   basic_plan_principal   = basic_plan.basic_plan()
percent_plan_invest, percent_plan_principal = percent_plan.percent_plan()
down_plan_invest,    down_plan_principal    = down_plan.down_plan()

fig, ax = plt.subplots()

ax.set_xlabel('month')
ax.set_ylabel('y')
ax.set_xlim([0, 121])
ax.set_ylim([0, 1000000])

month_x = np.linspace(0, 119, 120)
# ax.plot(month_x, price,                color='blue',   label='price')
ax.plot(month_x, basic_plan_invest,    color='orange', label='basic-invest')
ax.plot(month_x, basic_plan_principal, color='orange', label='basic-principal', linestyle='dotted')

ax.plot(month_x, percent_plan_invest,    color='green', label='percent-invest')
ax.plot(month_x, percent_plan_principal, color='green', label='percent-principal', linestyle='dotted')

ax.plot(month_x, down_plan_invest,    color='blue', label='down_invest')
ax.plot(month_x, down_plan_principal, color='blue', label='down-principal', linestyle='dotted')

ax.legend(loc=0)
plt.show()

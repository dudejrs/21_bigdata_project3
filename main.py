import streamlit as st
from streamlit import session_state as ss
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import datetime
import random
import pydeck as pdk


## 김준기님 import list

from bokeh.layouts import column, row, gridplot, layout
from bokeh.plotting import figure, show, gmap
from bokeh.io import output_file, show
from bokeh.models import ColumnDataSource, GMapOptions, DatetimeTickFormatter, NumeralTickFormatter
from bokeh.models import HoverTool, ColumnDataSource
from bokeh.models import Panel, Tabs #for Tab function
from bokeh.models import DateRangeSlider #for DateRangeSlider
from bokeh.models import CustomJS
from bokeh.tile_providers import get_provider, Vendors
from bokeh.tile_providers import CARTODBPOSITRON
from bokeh.models import WMTSTileSource
from bokeh.models import Div, RangeSlider, Spinner #for widget



st. set_page_config(layout="wide")
st.title('streamlit test')

bus_f = ['BUS_STATION_BOARDING_MONTH_201701.csv', 'BUS_STATION_BOARDING_MONTH_201702.csv', 'BUS_STATION_BOARDING_MONTH_201703.csv', 'BUS_STATION_BOARDING_MONTH_201704.csv', 'BUS_STATION_BOARDING_MONTH_201705.csv', 'BUS_STATION_BOARDING_MONTH_201706.csv', 'BUS_STATION_BOARDING_MONTH_201707.csv', 'BUS_STATION_BOARDING_MONTH_201708.csv', 'BUS_STATION_BOARDING_MONTH_201709.csv', 'BUS_STATION_BOARDING_MONTH_201710.csv', 'BUS_STATION_BOARDING_MONTH_201711.csv', 'BUS_STATION_BOARDING_MONTH_201712.csv', 'BUS_STATION_BOARDING_MONTH_201801.csv', 'BUS_STATION_BOARDING_MONTH_201802.csv', 'BUS_STATION_BOARDING_MONTH_201803.csv', 'BUS_STATION_BOARDING_MONTH_201804.csv', 'BUS_STATION_BOARDING_MONTH_201805.csv', 'BUS_STATION_BOARDING_MONTH_201806.csv', 'BUS_STATION_BOARDING_MONTH_201807.csv', 'BUS_STATION_BOARDING_MONTH_201808.csv', 'BUS_STATION_BOARDING_MONTH_201809.csv', 'BUS_STATION_BOARDING_MONTH_201810.csv', 'BUS_STATION_BOARDING_MONTH_201811.csv', 'BUS_STATION_BOARDING_MONTH_201812.csv', 'BUS_STATION_BOARDING_MONTH_201901.csv', 'BUS_STATION_BOARDING_MONTH_201902.csv', 'BUS_STATION_BOARDING_MONTH_201903.csv', 'BUS_STATION_BOARDING_MONTH_201904.csv', 'BUS_STATION_BOARDING_MONTH_201905.csv', 'BUS_STATION_BOARDING_MONTH_201906.csv', 'BUS_STATION_BOARDING_MONTH_201907.csv', 'BUS_STATION_BOARDING_MONTH_201908.csv', 'BUS_STATION_BOARDING_MONTH_201909.csv', 'BUS_STATION_BOARDING_MONTH_201910.csv', 'BUS_STATION_BOARDING_MONTH_201911.csv', 'BUS_STATION_BOARDING_MONTH_201912_1.csv']

@st.cache
def load_data():
    data1 = pd.read_csv("./data/pollution_weather.csv")
    data1["Measurement date"] = pd.to_datetime(data1["Measurement date"])
    data2 = pd.DataFrame([])

    for i in range(len(bus_f[:2])) :
        tmp = pd.read_csv("./data/bus_station_boarding/{}".format(bus_f[0]))
        tmp["사용일자"] = pd.to_datetime(tmp["사용일자"], format='%Y%m%d')
        data2 = data2.append(tmp)    


    return data1,data2

data_load_state = st.text('Loading data...')
data1_, data2_  = load_data()
data_load_state.text("Done! (using st.cache)")


st.sidebar.subheader('Contents')
contents = st.sidebar.selectbox(
    "choose Contents",
    ("A",'B',"C"),
    1
    )


st.sidebar.subheader('filter')

date_to_filter = st.sidebar.date_input("Choose Date",datetime.date(2017,1,1))
start_date_slider = st.sidebar.slider('Choose start Date', datetime.date(2017,1,1), datetime.date(2019,12,31),datetime.date(2017,1,1))
end_date_slider = st.sidebar.slider('Choose end Date',start_date_slider, datetime.date(2019,12,31),datetime.date(2017,12,31))

start_time = pd.to_datetime(datetime.datetime(start_date_slider.year,start_date_slider.month,start_date_slider.day))
end_time = pd.to_datetime(datetime.datetime(end_date_slider.year,end_date_slider.month,end_date_slider.day))


if 'region' not in ss : 
    ss.region = data1_['Address'].unique().tolist()
with st.sidebar.expander('Reigon') :
    ss.region = st.multiselect(' Choose region', data1_['Address'].unique().tolist(), data1_['Address'].unique().tolist())


data1 = data1_[(data1_["Measurement date"].dt.date >= start_date_slider) & (data1_["Measurement date"].dt.date <= end_date_slider) \
                & (data1_["Address"].isin(ss.region))\
]
data2 = data2_[(data2_["사용일자"].dt.date >= start_date_slider )&(data2_["사용일자"].dt.date <= end_date_slider) \
                & (data2_["name"].isin(ss.region))\
]



if contents == "A" :

##김준기_data_visualization-bokeh

    print("김준기_data_visualization-bokeh")
    # print(data1.loc[0,"Measurement date"])

    source1 =  ColumnDataSource(data={'x': data1["Measurement date"], 'y': data1["PM2.5"]})
    source1_1 =  ColumnDataSource(data={'x': data1["Measurement date"], 'y': data1["PM2.5"]})
    source2 =  ColumnDataSource(data={'x': data1["Measurement date"], 'y': data1["PM10"]})
    source2_1 =  ColumnDataSource(data={'x': data1["Measurement date"], 'y': data1["PM10"]})

    ### address0 - PM25 Plot ###
    #Interactive Tools#
    hover1 = HoverTool(tooltips=[('Timestamp', '@x'), ('초미세먼지', '@y')], formatters={'x': 'datetime'},)
    

    date_range_slider1 = DateRangeSlider(title="Test1", start=start_time, end=end_time,\
                                        value=(start_time, end_time), step=1, width=300)

    # date_range_slider1 = DateRangeSlider(title="Test1", start=start_time, end=end_time,\
    #                                     value=(start_time, end_time), step=1, width=300)

    # Plot #
    plot1 = figure(title = "종로구 초미세먼지", x_axis_label="측정날짜", y_axis_label="PM2.5",\
                   x_axis_type="datetime")

    plot1.line(x='x', y='y', source=source1, legend_label="PM2.5", line_width=2, color="blue", line_alpha=0.5)
    plot1.add_tools(hover1)

    callback1 = CustomJS(args=dict(source=source1, ref_source=source1_1), code="""
        
        console.log(cb_obj.value);
        
        const date_from = Date.parse(new Date(cb_obj.value[0]).toDateString());
        const date_to = Date.parse(new Date(cb_obj.value[1]).toDateString());
        
        const data = source.data;
        const ref = ref_source.data;
        
        const from_pos = ref["x"].indexOf(date_from);
        
        const to_pos = ref["x"].indexOf(date_to);
        
        data["y"] = ref["y"].slice(from_pos, to_pos);
        data["x"] = ref["x"].slice(from_pos, to_pos);
        
        source.change.emit();    
        """)

    date_range_slider1.js_on_change('value', callback1)
    layout1 = column(plot1, date_range_slider1)

    tab1 = Panel(child=layout1, title="test")



    ### address0 - PM10 Plot ###
    #Interactive Tools#
    hover2 = HoverTool(tooltips=[('Timestamp', '@x'), ('미세먼지', '@y')], formatters={'x': 'datetime'},)

    # Plot #
    plot2 = figure(title = "종로구 미세먼지", x_axis_label="측정날짜", y_axis_label="PM10",\
                x_axis_type="datetime")

    # Add to Plot #
    plot2.line(x='x', y='y', source=source2, legend_label="PM10", line_width=2, color="green", line_alpha=0.5)
    plot2.add_tools(hover2)


    callback2 = CustomJS(args=dict(source=source2, ref_source=source2_1), code="""
        
        console.log(cb_obj.value);
        
        const date_from = Date.parse(new Date(cb_obj.value[0]).toDateString());
        const date_to = Date.parse(new Date(cb_obj.value[1]).toDateString());
        
        const data = source.data;
        const ref = ref_source.data;
        
        const from_pos = ref["x"].indexOf(date_from);
        
        const to_pos = ref["x"].indexOf(date_to);
        
        data["y"] = ref["y"].slice(from_pos, to_pos);
        data["x"] = ref["x"].slice(from_pos, to_pos);
        
        source.change.emit();    
        """)
    date_range_slider1.js_on_change('value', callback2)
    layout2 = column(plot2, date_range_slider1)

    tab2 = Panel(child=layout2, title="test2")


    # p_bargraph.add_tools(HoverTool(renderers=[p25_bar],tooltips=tooltips1))

    plot1.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")
    plot2.xaxis[0].formatter = DatetimeTickFormatter(months="%b %Y")

    # show(Tabs(tabs=[tab1, tab2]))
    st.bokeh_chart(Tabs(tabs=[tab1,tab2]),use_container_width=True)

    print("hello")
    # st.bokeh_chart(Tabs([tab1))


elif contents == "B" :
    col1, col2, col3 = st.columns([5,4,1])
    
    if "B_pol" not in ss :
        ss.B_pol = 'SO2'
        ss.B_wtr = '기온(°C)'

    with col1 : 
        B_col1_region = st.multiselect('region',data1["Address"].unique().tolist(),data1["Address"].unique().tolist()[:2])
        B_data1 = data1[data1["Address"].isin(B_col1_region)]
        B_data2 = data2[data2["name"].isin(B_col1_region)]
        
        B_df = B_data1.groupby(['Address'], as_index= False).mean()

        st.pydeck_chart(\
             pdk.Deck(\
                map_style='mapbox://styles/mapbox/light-v9',\
             initial_view_state=pdk.ViewState(\
                     latitude=37.5720,\
                     longitude=127.0050,\
                     zoom=11,\
                     pitch=50,\
                 ),\
                 layers=[\
                     pdk.Layer(\
                        'ColumnLayer',\
                        data=B_data1.loc[:,['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5','Longitude','Latitude']],\
                        get_position='[Longitude,Latitude]',\
                        radius=200,\
                        get_elevation=[ss.B_pol],\
                        elevation_scale=10000,\
                        get_fill_color=["S02 * 256","NO2 * 256", "CO * 256"],\
                        pickable=True,\
                     )\
                     # ,pdk.Layer(\
                     #    'ScatterplotLayer',\
                     #    data=B_data1.loc[:,['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5','Longitude','Latitude']],\
                     #    get_position='[Longitude,Latitude]',\
                     #    radius=200,\
                     #    elevation_scale=100,\
                     #    elevation_range=[0, 1000],\
                     #    pickable=True,\
                     #    extruded=True,\
                     # )\
                 ],\
            )
        )

        # st.dataframe(B_data1)
        st.dataframe(B_df)
        st.dataframe(B_data2)

        


    with col2 :
        with st.container() :
            B_start = st.slider('Cs', datetime.date(2017,1,1), end_date_slider)
            B_end = B_start + timedelta(weeks=2)


        fig, ax = plt.subplots(2,1)
        st.subheader("{}".format(ss.B_pol))

        sns.lineplot(data= B_data1[ (B_data1["Measurement date"].dt.date >= B_start) & (B_data1["Measurement date"].dt.date <= B_end)],\
            x="Measurement date", y = ss.B_pol, hue= "Address" , ax= ax[0])
        ax[0].set(xticklabels=[], xlabel="", ylabel="",title="{}".format(ss.B_pol))
        ax[1].set(xticklabels=[], xlabel="", ylabel="",title="{}".format(ss.B_wtr))


        sns.lineplot(data= B_data1[ (B_data1["Measurement date"].dt.date >= B_start) & (B_data1["Measurement date"].dt.date <= B_end)],\
            x="Measurement date", y = ss.B_wtr, ax= ax[1])

        st.pyplot(fig)


        df = B_data2.groupby(['사용일자','name'],as_index=False,).sum().loc[:,["사용일자","name","승차총승객수","하차총승객수"]]
        df['승하차승객수'] = df['승차총승객수'] + df['하차총승객수']
        df = df.loc[:,['사용일자','name','승하차승객수']].dropna()
        # df = df.pivot(index='사용일자',columns='name',values='승하차승객수')


        st.subheader("승하차승객수")
        plt.clf()
        ax = sns.barplot(x="사용일자",y="승하차승객수",hue="name",data=df)
        ax.set(xticklabels=[], xlabel="", ylabel="")
        st.pyplot(ax.figure)
        # st.bar_chart(df["승차총승객수"], use_container_width = True)

    with col3 :

        st.subheader("pollution")
        ss.B_pol = st.radio("Chose pollution", ['SO2', 'NO2', 'O3', 'CO', 'PM10', 'PM2.5'])
        st.subheader("날씨")
        ss.B_wtr = st.radio("Chose pollution", ['기온(°C)','강수량(mm)', '풍속(m/s)', '풍향(16방위)', '습도(%)', '현지기압(hPa)', '해면기압(hPa)','일조(hr)', '일사(MJ/m2)', '적설(cm)'])

    print(ss.B_pol, ss.B_wtr)        


else :
    print("C")



    

    











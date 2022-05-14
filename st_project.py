import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import altair as alt
import time
import squarify
from bokeh.plotting import figure
from bokeh.models import NumeralTickFormatter
from bokeh.transform import factor_cmap, factor_mark

with st.echo(code_location='below'):
    st.title('Data Science Mini Project')
    st.text('Работа делалась по ночам и предполагает использование тёмной темы =)')
    st.sidebar.markdown("""
            # Sections
            - [Data Distribution](#distribution)
            - [Ranking](#ranking)
            - [Show correlation with Means](#mean_corr)
            - [Show correlation with Filtration](#filter_corr)
            - [Animation](#animation)
            """, unsafe_allow_html=True)
    st.header("Preparing Data")
    st.text('Данные по вакансиям на должности, связанные с Data Science')


    # Preparing Data
    @st.cache
    def get_dataframe():
        df = pd.read_csv(r'data_cleaned_2021.csv').drop(columns='index').drop_duplicates().drop(
            columns=['Job Description', 'Job Title'])
        rename_dict = {'seniority_by_title': 'seniority',
                       'Avg Salary(K)': "avg_salary",
                       'company_txt': 'company',
                       'job_title_sim': 'job',
                       'Revenue': 'company_revenue',
                       "Type of ownership": 'own_type',
                       'Size': 'company_size'}
        size_list = ['1 - 50 ', '51 - 200 ', '201 - 500 ', '501 - 1000 ', '1001 - 5000 ', '5001 - 10000 ', '10000+ ']
        new_size_list = ['1-50', '51-200', '201-500', '501-1000', '1001-5000', '5001-10000', '10000+']
        revenue_list = [
            'Less than $1 million (USD)', '$1 to $5 million (USD)',
            '$5 to $10 million (USD)', '$10 to $25 million (USD)',
            '$25 to $50 million (USD)', '$50 to $100 million (USD)', '$100 to $500 million (USD)',
            '$500 million to $1 billion (USD)', '$1 to $2 billion (USD)',
            '$2 to $5 billion (USD)', '$5 to $10 billion (USD)',
            '$10+ billion (USD)', 'Unknown / Non-Applicable']
        new_revenue_list = ['NA', '<$1M', '$1M-$5M', '$5M-$10M', '$10M-$25M', '$25M-$50M', '$50M-$100M',
                            '$100M-$500M', '$500M-$1B', '$1B-$2B', '$2B-$5B', '$5-$10B', '$10B+']
        df = df.rename(columns=rename_dict).replace(size_list, new_size_list)
        df.columns = df.columns.str.lower()
        df.seniority.replace(['na', 'jr', 'sr'], ['Other', 'Other', 'Senior'], inplace=True)
        df.degree.replace(['na', 'M', 'P'], ['Not Available', 'Master', 'PhD'], inplace=True)
        df.company_revenue.replace(revenue_list, new_revenue_list, inplace=True)
        df = df.assign(skill_num=lambda x: sum([x[a] for a in skills]))
        df = df[(df.sector != '-1') * (df.industry != '-1') * (df['company_size'] != 'unknown') * (df.rating >= 0) * (
                df.founded != -1)]
        return df


    @st.cache
    def get_aggregated_by(param):
        def f(x):
            def get_several_lined_string(series):
                lst = []
                for f_i in range(0, len(series), 5):
                    lst.append(', '.join(series[f_i:8 + f_i]))
                return ',\n'.join(lst)

            jobs_txt = get_several_lined_string(x.job.unique())
            ### FROM: https://stackoverflow.com/questions/17841149/pandas-groupby-how-to-get-a-union-of-strings
            dct = {skill: round(x[skill].mean(), 3) for skill in skills}
            dct['jobs'] = jobs_txt
            dct['skill_num'] = round(x['skill_num'].mean(), 3)
            ### END FROM
            return pd.Series(dct)

        param_agg = wage_df.groupby(param).apply(f).reset_index()
        param_agg = (param_agg[param_agg.skill_num > 0].iloc[:, :-1]
                     .melt(id_vars=[param, 'jobs'])
                     .rename(columns={'variable': 'skill', 'value': 'share'}))
        return param_agg


    def get_data_for_ranking(name, highlight, over):
        agg_dict = {'industry': industry_agg, 'sector': sector_agg, 'skill': sector_agg}
        agg = agg_dict[over]
        if over == 'skill':
            skill_or_sector = 'sector'
            label = f'Частота использования {name.capitalize()}'
        else:
            skill_or_sector = 'skill'
            label = f'Частота использования в {name}'
        df = agg[agg[over] == name].sort_values(by='share', ascending=False)
        highlight_total = (np.array([df[skill_or_sector] ==
                                     hl for hl in highlight]).sum(axis=0)
                           if highlight != []
                           else np.ones(df[skill_or_sector].shape))
        color = np.where(highlight_total, '#F66B0E', '#205375')
        sizes = np.where(highlight_total, 300, 150)
        ranking_dict = {'df': df,
                        'skill_or_sector': skill_or_sector,
                        'label': label, 'color': color, 'sizes': sizes}
        return ranking_dict


    def get_data_for_corr(param):
        x1, y1, x2, y2, order, x_label = 0, 0, 0, 0, [], ''
        if param == 'skill_num':
            x1, y1, x2, y2 = 1.1, 127, 0.2, 88.5
            order = list(range(11))
            x_label = 'Number of skills from the list'
        elif param == 'company_revenue':
            x1, y1, x2, y2 = 1.1, 170, 0.2, 120
            order = ['NA', '<$1M', '$1M-$5M', '$5M-$10M', '$10M-$25M', '$25M-$50M', '$50M-$100M',
                     '$100M-$500M', '$500M-$1B', '$1B-$2B', '$2B-$5B', '$5-$10B', '$10B+']
            x_label = 'Company Revenue'
        elif param == 'company_size':
            x1, y1, x2, y2 = 0.3, 150, 0.2, 105
            order = ['1-50', '51-200', '201-500', '501-1000', '1001-5000', '5001-10000', '10000+']
            x_label = 'Company Size'
        elif param == 'rating':
            x1, y1, x2, y2 = 4.8, 50, 2.35, 112
            order = sorted(wage_df.rating.unique())
            x_label = 'Company Rating'
        angle = -20 if param == 'company_revenue' else 0
        data_type = 'quantitative' if param in ['rating', 'skill_num'] else 'ordinal'
        return dict(x1=x1, x2=x2, y1=y1, y2=y2, order=order, x_label=x_label, angle=angle, type=data_type)


    def customize_axis(axis):
        sns.set(rc={'axes.facecolor': '#0e1117', 'figure.facecolor': '#0e1117'})
        axis.xaxis.label.set_color('#EFEFEF')
        axis.yaxis.label.set_color('#EFEFEF')
        axis.tick_params(axis='x', colors='#EFEFEF')
        axis.grid(visible=True, axis='x', color='#205375')
        axis.grid(visible=True, axis='y', color='#205375')


    def make_radio_horizontal():
        ### FROM: https://discuss.streamlit.io/t/horizontal-radio-buttons/2114/8
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;} </style>',
                 unsafe_allow_html=True)
        st.write('<style>div.st-bf{flex-direction:column;} div.st-ag{font-weight:bold;padding-left:2px;}</style>',
                 unsafe_allow_html=True)
        ### END FROM


    def get_categorical_parameter(user_input):
        parameter = None
        if user_input == 'Number of skills':
            parameter = 'skill_num'
        elif user_input == 'Company Revenue':
            parameter = 'company_revenue'
        elif user_input == 'Company Size':
            parameter = 'company_size'
        elif user_input == 'Company Rating':
            parameter = 'rating'
        elif user_input == 'Foundation Year':
            parameter = 'founded'
        elif user_input == 'Job Title':
            parameter = 'job'
        elif user_input == 'Degree':
            parameter = 'degree'
        elif user_input == 'Ownership':
            parameter = 'own_type'
        return parameter


    # Data Distribution
    skills = ['python', 'spark', 'aws', 'excel', 'sql', 'sas', 'keras',
              'pytorch', 'scikit', 'tensor', 'hadoop', 'tableau',
              'bi', 'flink', 'mongo', 'google_an']
    wage_df = get_dataframe()
    industry_agg = get_aggregated_by('industry')
    sector_agg = get_aggregated_by('sector')
    st.dataframe(wage_df)

    st.header('Distributions of data in  the dataset', anchor='distribution')
    st.text('Note that companies in records are not unique')
    st.text('Plot types: TreeMap, круговая, гистограмма')

    make_radio_horizontal()
    dist_radio = st.selectbox("Please, select a type of data",
                              ('Company Revenue', 'Foundation Year', 'Number of skills', 'Company Rating',
                               'Company Size', 'Job Title', 'Degree', 'Ownership'))
    dist_param = get_categorical_parameter(dist_radio)
    st.text(dist_param)
    if dist_param in ['rating', 'skill_num']:
        fig, dist_axis = plt.subplots()
        sns.countplot(data=wage_df, x=dist_param,
                      palette=sns.dark_palette("#69d", n_colors=len(wage_df[dist_param].unique())))
        customize_axis(dist_axis)
        dist_axis.tick_params(axis='y', colors='#EFEFEF')
        for i, label in enumerate(dist_axis.xaxis.get_ticklabels()):
            label.set_visible(False)
            if dist_param == 'founded':
                if i % 30 == 0:
                    label.set_visible(True)
            else:
                if i % 2 == 0:
                    label.set_visible(True)
        st.pyplot(fig)
    elif dist_param == 'founded':
        dist_chart = alt.Chart(wage_df).mark_bar().encode(
            alt.X('founded:Q', axis=alt.Axis(format="c")), alt.Y('count(founded):Q'), color=alt.value('#205375')
        ).properties(width=700, height=350)
        st.altair_chart(dist_chart.interactive())
    else:
        fig, dist_axis = plt.subplots()
        type_radio = st.radio("Please, select a plot type",
                              ('Pie Chart (matplotlib)', 'TreeMap (squarify)'))
        sizes = wage_df[dist_param].value_counts().sort_values() / wage_df[dist_param].value_counts().sum() * 100
        sizes = sizes.where(sizes > 4.9, float('nan')).dropna()
        if sum(sizes) < 99.9:
            sizes = pd.concat([sizes, pd.Series({'Other': round(100 - sizes.sum(), 2)})])
        if type_radio == 'TreeMap (squarify)':

            squarify.plot(sizes=sizes, label=sizes.index, value=sizes.round(1), alpha=0.8,
                          color=sns.color_palette('muted'))
            dist_axis.axis('off')
        else:
            patches, texts, pcts = dist_axis.pie(sizes, labels=sizes.index, autopct='%1.1f%%',
                                                 colors=sns.color_palette('muted'), startangle=90)
            ### FROM: https://www.pythoncharts.com/matplotlib/pie-chart-matplotlib/
            for i, patch in enumerate(patches):
                texts[i].set_color(patch.get_facecolor())
            ### END FROM
        st.pyplot(fig)

    # Ranking
    st.header('Ranking')
    st.text('Plot types: леденцовая, линейчатая')


    def bokeh_customize(plot):
        plot.xaxis.axis_label_text_color = "#EFEFEF"
        plot.xgrid.grid_line_color = '#205375'
        plot.xaxis.major_label_text_color = "#EFEFEF"
        plot.xaxis.axis_line_color = "#EFEFEF"

        plot.yaxis.axis_label_text_color = "#EFEFEF"
        plot.ygrid.grid_line_color = None
        plot.yaxis.axis_line_color = "#EFEFEF"
        plot.yaxis.major_label_text_color = "#EFEFEF"

        plot.title.text_color = "#EFEFEF"
        plot.background_fill_color = '#0e1117'
        plot.border_fill_color = '#0e1117'


    def sns_ranking(name, highlight, over):
        highlight = highlight if highlight else []
        data = get_data_for_ranking(name, highlight, over)
        sns.set_theme(context='notebook', style='dark')
        sns.set(rc={'axes.facecolor': '#0e1117', 'figure.facecolor': '#0e1117'})
        sns.barplot(data=data['df'], x='share', y=data['skill_or_sector'], palette=data['color'],
                    linewidth=0).set(xlabel=data['label'], ylabel=data['skill_or_sector'].capitalize())
        for i, color in enumerate(data['color']):
            color = color if color == '#F66B0E' else '#EFEFEF'
            plt.gca().get_yticklabels()[i].set_color(color)


    def alt_ranking(name, highlight, over):
        data = get_data_for_ranking(name, highlight, over)
        condition_bars = alt.condition(
            alt.FieldOneOfPredicate(data['skill_or_sector'], highlight),
            alt.value('#F66B0E'),
            alt.value('#205375'))
        condition_text = (
            alt.condition(
                alt.FieldOneOfPredicate(data['skill_or_sector'], highlight),
                alt.value('#F66B0E'),
                alt.value('#EFEFEF')
            )
        )
        x = alt.X('share',
                  axis=alt.Axis(labelColor='#EFEFEF', titleColor='#EFEFEF',
                                labelFontSize=16, titleFontSize=20, format='%',
                                title=data['label']), scale=alt.Scale(domain=[0, 1]))
        y = alt.Y(data['skill_or_sector'], sort='-x',
                  axis=alt.Axis(labelColor='#EFEFEF', titleColor='#EFEFEF',
                                labelFontSize=12, titleFontSize=20,
                                title=data['skill_or_sector'].capitalize()))
        rank_chart = alt.Chart(data['df']).mark_bar().encode(
            y=y, x=x, color=condition_bars,
            tooltip=[alt.Tooltip('jobs', title='Jobs in Sector/Industry')]
        ).properties(width=750, height=500)
        text = rank_chart.mark_text(
            align='left', baseline='middle', dx=5
        ).encode(
            text=alt.Text('share', format='.1%'), color=condition_text
        )
        return (rank_chart + text).configure(background='#0e1117')


    def plt_ranking(name, highlight, over):
        plt.rcParams.update({'font.size': 19
                             })
        data = get_data_for_ranking(name, highlight, over)
        df = data['df'].iloc[::-1]
        skill_or_sector = data['skill_or_sector']
        ### FROM:'https://python-graph-gallery.com/183-highlight-a-group-in-lollipop'
        plt.hlines(y=df[skill_or_sector], xmin=0, xmax=df.share, color=np.flip(data['color']), alpha=1, linewidth=4)
        plt.scatter(x=df.share, y=df[skill_or_sector], color=np.flip(data['color']), s=np.flip(data['sizes']), alpha=1)
        ### END FROM
        plt.xlabel(data['label'])
        plt.ylabel(skill_or_sector.capitalize())
        for i, color in enumerate(np.flip(data['color'])):
            color = color if color == '#F66B0E' else '#EFEFEF'
            plt.gca().get_yticklabels()[i].set_color(color)


    def bokeh_ranking(name, highlight, over):
        highlight = highlight if highlight else []
        data = get_data_for_ranking(name, highlight, over)
        df = data['df'].iloc[::-1]
        df['color'] = data['color'][::-1]
        skill_or_sector = data['skill_or_sector']

        p = figure(y_range=df[skill_or_sector], plot_width=700, plot_height=500, x_range=(0, 1), title=name,
                   outline_line_color="#EFEFEF", tools='hover', tooltips=[('Jobs', '@jobs'), ('share', '@share')])
        p.hbar(y=skill_or_sector, right='share', height=0.8, fill_color='color', line_color=None, source=df)

        p.xaxis.axis_label = data['label']
        p.xaxis[0].formatter = NumeralTickFormatter(format="0%")
        p.yaxis.axis_label = data['skill_or_sector'].capitalize()
        bokeh_customize(p)

        return p


    ranking = st.radio("Choose ranking type", ('Skills over sector', 'Skills over industry', 'Sectors over skill'))

    if ranking == 'Skills over sector':
        selector = st.selectbox("Sector", np.sort(sector_agg.sector.unique()), index=1)
        highlight = st.multiselect("What skills would you like to highlight?", skills, default='python')
        over = 'sector'
    elif ranking == 'Skills over industry':
        selector = st.selectbox("Industry", np.sort(industry_agg.industry.unique()))
        highlight = st.multiselect("What skills would you like to highlight?", skills)
        over = 'industry'
    elif ranking == 'Sectors over skill':
        selector = st.selectbox("Skill", skills)
        highlight = st.multiselect("What sector would you like to highlight?", wage_df.sector.unique())
        over = 'skill'
    make_radio_horizontal()
    rank_lib = st.radio("Plotting library", ("Seaborn", "Altair", 'Bokeh', "Matplotlib"))

    if rank_lib == 'Seaborn':
        rank_plot, ax = plt.subplots(figsize=(8, 8))
        sns_ranking(selector, highlight, over)
        customize_axis(ax)
        plt.xlim([0, 1])
        st.pyplot(rank_plot, True)
    elif rank_lib == 'Altair':
        alt_rank_chart = alt_ranking(selector, highlight, over)
        st.altair_chart(alt_rank_chart.interactive())
    elif rank_lib == 'Matplotlib':
        rank_plot, ax = plt.subplots(figsize=(8, 8))
        plt_ranking(selector, highlight, over)
        customize_axis(ax)
        plt.xlim([0, 1])
        st.pyplot(rank_plot, True)
    elif rank_lib == 'Bokeh':
        rank_plot = bokeh_ranking(selector, highlight, over)
        st.bokeh_chart(rank_plot)

    # Correlation
    def sns_wage_corr(param, corr_plot_type):
        data = get_data_for_corr(param)
        sns.set_theme(context='notebook', style='dark')
        sns.set(rc={'axes.facecolor': '#0e1117', 'figure.facecolor': '#0e1117'})
        if corr_plot_type == 'Scatter Plot':
            sns.stripplot(
                data=wage_df, x=param, y="avg_salary", hue="seniority",
                dodge=False, order=data['order'], alpha=0.4)
            sns.pointplot(
                data=wage_df,
                x=param,
                y='avg_salary',
                hue='seniority',
                order=data['order'],
                hue_order=['Other', 'Senior'],
                scale=0.9
            ).set(xlabel=data['x_label'], ylabel='Salary (K)')
            ax.tick_params(axis='y', colors='#EFEFEF')
            plt.xticks(rotation=-data['angle'])
            plt.text(data['x1'], data['y1'], "Senior", horizontalalignment='left', size='medium', color='#da8251',
                     weight='semibold')
            plt.text(data['x2'], data['y2'], "Other", horizontalalignment='left', size='medium', color='#4c72b0',
                     weight='semibold')
        else:
            order = data['order'][::-1] if param in ('Company Revenue', 'Company Size') else wage_df[param].unique()
            sns.boxplot(data=wage_df, x='avg_salary', y=param, order=order
                        ).set(ylabel=param, xlabel='Salary (K)')
            ax.tick_params(axis='y', colors='#EFEFEF')



    def alt_wage_corr(param, corr_plot_type):
        data = get_data_for_corr(param)
        if corr_plot_type == 'Scatter Plot':
            x = alt.X(param, sort=data['order'], type=data['type'],
                      axis=alt.Axis(labelAngle=data['angle'], labelColor='#EFEFEF', titleColor='#EFEFEF',
                                    labelFontSize=16, titleFontSize=20, title=data['x_label'])
                      )
            y = alt.Y('avg_salary', axis=alt.Axis(labelColor='#EFEFEF', titleColor='#EFEFEF',
                                                  labelFontSize=12, titleFontSize=20, title='Salary (K)')
                      )
            base = alt.Chart(wage_df).mark_circle(opacity=0.5).encode(
                x, y,
                alt.Color('seniority', legend=alt.Legend(
                    title='Seniority', orient="top")),
                tooltip=['company', 'company_revenue', 'avg_salary', 'job', 'company_size', 'own_type', param]
            )

            if param in ['rating', 'skill_num']:
                line = base.transform_loess(param, 'avg_salary', groupby=['seniority']).mark_line()
            else:
                key = dict(zip(data['order'], range(len(data['order']))))
                source = wage_df.assign(order=lambda x: x[param].map(key))
                line = alt.Chart(source).mark_line().encode(
                    x=alt.X('order', axis=None),
                    y=alt.Y('mean(avg_salary)'),
                    color=alt.Color('seniority')
                )

            return (base + line).configure(background='#0e1117').properties(width=750, height=500)
        else:
            order = data['order'][::-1] if param in ('Company Revenue', 'Company Size') else wage_df[param].unique()
            y = alt.Y(f'{param}:O', sort=order,
                      axis=alt.Axis(labelAngle=0, labelColor='#EFEFEF', titleColor='#EFEFEF',
                                    labelFontSize=16, titleFontSize=20)
                      )
            x = alt.X('avg_salary:Q', axis=alt.Axis(labelColor='#EFEFEF', titleColor='#EFEFEF',
                                                  labelFontSize=12, titleFontSize=20)
                      )
            boxes = alt.Chart(wage_df).mark_boxplot(size=50, outliers=True).encode(y=y, x=x,
                                                                                   color=alt.Color(param, legend=None))
            return boxes.configure(background='#0e1117').properties(width=750, height=700)

    def bokeh_wage_corr(param):
        data = get_data_for_corr(param)

        # create column for sorting
        key = dict(zip(data['order'], range(len(data['order']))))
        offset = -0.5 if param == 'company_revenue' else 0.5
        source = wage_df.assign(order=lambda x: x[param].map(key) + offset).sort_values(by='order')

        source.avg_salary *= 1000
        tooltips = [('Company Name', '@company'),
                    ('Company Revenue', '@company_revenue'),
                    ('Salary', '@avg_salary'),
                    ('Job Title', '@job'),
                    ('Company Size', '@company_size'),
                    ('Ownership type', '@own_type'),
                    ('Company Rating', '@rating')
                    ]

        p = figure(plot_width=700, plot_height=500, title="Salary correlation", background_fill_color='#0e1117',
                   x_range=source[param].unique().astype('str'), tools='hover', tooltips=tooltips)

        # scatter plot
        p.scatter('order', "avg_salary", source=source,
                  legend_group="seniority", fill_alpha=0.1, size=12,
                  marker=factor_mark('seniority', ['hex', 'triangle'], source.seniority.unique()),
                  color=factor_cmap('seniority', 'Category10_3', source.seniority.unique()))

        # mean lines
        mean_df_senior = source[source.seniority == 'Senior'].groupby(param).mean().reset_index().sort_values(
            by='order')
        mean_df_other = source[source.seniority == 'Other'].groupby(param).mean().reset_index().sort_values(
            by='order')
        p.line('order', 'avg_salary', line_width=3, source=mean_df_senior, line_color='#da8251')
        p.line('order', 'avg_salary', line_width=3, source=mean_df_other, line_color='#4c72b0')

        # customization
        bokeh_customize(p)
        p.xaxis.major_label_orientation = np.pi / 6 if param == 'company_revenue' else 'horizontal'
        p.xaxis.axis_label = data['x_label']
        p.xgrid.grid_line_color = None
        p.yaxis.axis_label = 'Salary'
        p.ygrid.grid_line_color = '#205375'
        p.yaxis[0].formatter = NumeralTickFormatter(format='($0 a)')
        return p


    st.header('Show correlation with Means', anchor='mean_corr')
    st.text('Plot types: график рассеяния с линией тренда, ящик с усами')

    corr_plot_type = st.radio("Choose the desired type of plot:",
                    ('Scatter Plot', 'Box Plot'))
    if corr_plot_type == 'Scatter Plot':
        corr_options = ('Number of skills', 'Company Revenue', 'Company Size', 'Company Rating')
        corr_libs_list = ("Seaborn", "Altair", "Bokeh")
    else:
        corr_options = ('Company Revenue', 'Company Size', 'Job Title', 'Degree', 'Ownership')
        corr_libs_list = ("Seaborn", "Altair")

    corr_radio = st.radio("Choose parameter to correlate:", corr_options)
    corr_parameter = get_categorical_parameter(corr_radio)
    corr_lib = st.radio("Plotting library:", corr_libs_list)

    if corr_lib == 'Seaborn':
        fig, ax = plt.subplots(figsize=(10, 5))
        sns_wage_corr(corr_parameter, corr_plot_type)
        customize_axis(ax)
        ax.legend_ = None
        if corr_parameter == 'rating':
            for label in ax.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
        ax.autoscale()
        st.pyplot(fig, True)
    elif corr_lib == 'Altair':
        alt_corr_chart = alt_wage_corr(corr_parameter, corr_plot_type)
        st.altair_chart(alt_corr_chart.interactive())
    elif corr_lib == 'Bokeh':
        st.bokeh_chart(bokeh_wage_corr(corr_parameter))


    # Filtered Correlation
    def alt_corr_filtered(scatter_param='company_size', hist_param='company_revenue'):
        brush = alt.selection_interval(encodings=['x'])
        multi = alt.selection_multi(encodings=['x'])

        scatter_data = get_data_for_corr(scatter_param)
        scatter_order, scatter_angle = scatter_data['order'], scatter_data['angle']

        key = dict(zip(scatter_data['order'], range(len(scatter_data['order']))))
        source = wage_df.assign(order=lambda x: x[scatter_param].map(key))
        lst = []
        ### FROM: https://stackoverflow.com/questions/68841230/how-to-replace-the-axis-label-in-altair
        # Скорее основано, но всё же
        for i, label in zip(range(len(scatter_data['order'])), scatter_data['order']):
            lst.append(f"datum.label == {i} ? '{label}'")
        lst.append("''")
        axis_labels = ' : '.join(lst)
        ### END FROM
        scatter_x = alt.X('order', scale=alt.Scale(domain=[-1, len(scatter_data['order'])]),
                          axis=alt.Axis(labelAngle=scatter_angle, labelColor='#EFEFEF', titleColor='#EFEFEF',
                                        labelFontSize=14, titleFontSize=20, tickMinStep=1, ticks=True, labelOverlap=False,
                                        title=scatter_data['x_label'], orient='top', labelExpr=axis_labels))

        ### FROM: https://altair-viz.github.io/gallery/scatter_with_histogram.html
        scatter = alt.Chart().mark_circle(opacity=0.5).encode(
            x=scatter_x,
            y=alt.Y('avg_salary', scale=alt.Scale(domain=[0, 260]),
                    axis=alt.Axis(labelColor='#EFEFEF', titleColor='#EFEFEF',
                                  labelFontSize=12, titleFontSize=20, title='Average Salary (K)')),
            color=alt.Color('seniority', legend=alt.Legend(title='Seniority', orient="right")),
            tooltip=['company', 'company_revenue', 'avg_salary', 'job', 'company_size', 'own_type']
        ).transform_filter(
            brush | multi
        ).properties(width=800, height=300)
        ###END FROM
        reg = scatter.transform_regression('order', 'avg_salary', groupby=['seniority']).mark_line()
        hist_data = get_data_for_corr(hist_param)
        hist_order, hist_angle = hist_data['order'], hist_data['angle']
        hist_x = alt.X(hist_param, sort=hist_order, type='ordinal',
                       axis=alt.Axis(labelAngle=-20, labelColor='#EFEFEF', titleColor='#EFEFEF',
                                     labelFontSize=16, titleFontSize=20,
                                     title=hist_data['x_label']
                                     ))
        hist = alt.Chart().mark_bar().encode(x=hist_x,
                                             y=alt.Y("count()",
                                                     axis=alt.Axis(labelColor='#EFEFEF', titleColor='#EFEFEF',
                                                                   labelFontSize=12, titleFontSize=20,
                                                                   title='Count of Records')),
                                             color=alt.condition(brush | multi, alt.value('#F66B0E'),
                                                                 alt.value('#205375'))
                                             ).properties(width=800, height=300).add_selection(brush, multi)

        try:
            total_chart = alt.vconcat(
                (scatter + reg),
                hist,
                data=source)
            return total_chart.configure(background='#0e1117')
        except:
            pass


    st.header('Show Correlation with filtration', anchor='filter_corr')
    st.text('Plot types: график рассеяния со средним, гистограмма')
    with st.expander("Hint"):
        st.subheader('Filtration_mechanism:')
        st.text("""
                Select an area on the hist plot. 
                The corresponding information will be showed on scatter plot.
                You can also select individual column by shift+click. 
                Note that only one type of selection can be applied
                """)
        st.subheader('Interpretation:')
        st.text("""
                Red line shows mean over filtered data. Gray line shows the mean of all records.
                If the red line is higher than the gray one, the filtration positively 
                affects the mean and vise-versa.
                """)
    scatter_radio = st.radio("Please, select a parameter to correlate with salary",
                             ('Number of skills', 'Company Revenue', 'Company Size', 'Company Rating'))
    hist_radio = st.radio('Please, select a parameter to filter the information',
                          ('Number of skills', 'Company Revenue', 'Company Size', 'Company Rating'))
    alt_filter_chart = alt_corr_filtered(get_categorical_parameter(scatter_radio),
                                         get_categorical_parameter(hist_radio))
    st.altair_chart(alt_filter_chart)

    # Animation
    st.header('Animation')
    st.text('Plot types: гистограмма')
    st.text('Shows the amount of records within each category as salary threshold increases')
    st.text('Анимация нужна, чтобы лучше увидеть динамику.\n '
            'Например, при переходе от порога $75K к порогу $90K можно заметить резкий скачок\n'
            'Также видно, что ваансий с зарплатами выше $200K почти нет')
    anime_radio = st.radio("Please, select a parameter to correlate with salary",
                           ('Number of skills', 'Company Revenue', 'Company Size', 'Company Rating'), key='anime_radio')
    anime_param = get_categorical_parameter(anime_radio)
    anime_lib = st.radio('Please, select a library to use:', ('Altair', 'Bokeh'))
    max_y = max(wage_df[anime_param].value_counts())


    def alt_animation(df, param=anime_param, plot_opacity=1.0):
        data = get_data_for_corr(param)
        anime_order, anime_angle = data['order'], data['angle']
        anime_x = alt.X(param, sort=anime_order, type='ordinal',
                        axis=alt.Axis(labelAngle=anime_angle, labelColor='#EFEFEF', titleColor='#EFEFEF',
                                      labelFontSize=16, titleFontSize=20,
                                      title=data['x_label']))
        anime_y = alt.Y("count()", scale=alt.Scale(domain=[0, max_y]),
                        axis=alt.Axis(labelColor='#EFEFEF', titleColor='#EFEFEF',
                                      labelFontSize=12, titleFontSize=20,
                                      title='Count of Records'))
        anime_chart = alt.Chart(df).mark_bar(opacity=plot_opacity).encode(x=anime_x, y=anime_y
                                                                          ).properties(width=600, height=300)
        return anime_chart


    def bokeh_animation(df, param=anime_param):
        data = get_data_for_corr(param)

        key = dict(zip(data['order'], range(len(data['order']))))
        offset = -0.5 if param == 'company_revenue' else 0.5
        counts = (df[param].value_counts(sort=False)
                  .reset_index()
                  .assign(order=lambda x: x['index'].map(key) + offset)
                  .sort_values(by='order')
                  )
        counts_total = (wage_df[param].value_counts(sort=False)
                        .reset_index()
                        .assign(order=lambda x: x['index'].map(key) + offset)
                        .sort_values(by='order')
                        )
        p = figure(x_range=counts_total['index'].unique().astype('str'), plot_width=700, plot_height=500,
                   title=data['x_label'],
                   outline_line_color="#EFEFEF")
        p.vbar(x=counts['order'], top=counts[param], width=0.8, line_color=None, fill_alpha=1.0)
        p.vbar(x=counts_total['order'], top=counts_total[param], width=0.8, line_color=None, fill_alpha=0.3)
        p.xaxis.axis_label = data['x_label']
        p.xaxis.major_label_orientation = np.pi / 6 if param == 'company_revenue' else 'horizontal'
        p.yaxis.axis_label = 'Count of Records'
        bokeh_customize(p)
        return p


    slider_threshold = st.slider("Salary range", 0, 255, (100, 200))
    slider_df = wage_df[(wage_df.avg_salary > slider_threshold[0]) * (wage_df.avg_salary <= slider_threshold[1])]

    shadow_plot = alt_animation(wage_df, plot_opacity=0.3)
    slider_plot = alt_animation(slider_df) + shadow_plot if anime_lib == 'Altair' else bokeh_animation(slider_df)

    anime_plot = st.altair_chart(slider_plot + shadow_plot) if anime_lib == 'Altair' else st.bokeh_chart(slider_plot)
    status_text = st.empty()
    status_text.text(f'Salary Range = ${slider_threshold[0]}K-${slider_threshold[1]}K')


    def start_animation():
        global anime_plot, status_text, shadow_plot, slider_threshold
        ### FROM: https://towardsdatascience.com/how-to-run-animations-in-altair-and-streamlit-2a0624789ad
        for threshold in range(0, 255, 15):
            status_text.text(f'Salary threshold = ${threshold}K')
            step_df = wage_df[wage_df.avg_salary <= threshold]
            anime_chart = alt_animation(step_df) + shadow_plot if anime_lib == 'Altair' else bokeh_animation(step_df)
            anime_plot = anime_plot.altair_chart(anime_chart) if anime_lib == 'Altair' else anime_plot.bokeh_chart(
                anime_chart)
            time.sleep(0.7)
        ### END FROM


    start_btn = st.button('Start', on_click=start_animation)

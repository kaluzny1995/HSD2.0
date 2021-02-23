import twint

from datetime import timedelta

from .utils.dates import daterange


def amount_from_each_day(start, end, topic, lang='en', limit=None, file_name=None):
    c = twint.Config()
    for single_date in daterange(start, end):
        c.Since = single_date.strftime("%Y-%m-%d")
        c.Until = (single_date + timedelta(1)).strftime("%Y-%m-%d")
        c.Search = topic
        c.Lang = lang
        if limit:
            c.Limit = limit
        if file_name:
            c.Store_csv = True
            c.Output = file_name

        twint.run.Search(c)
        print('Scraping from {} completed!'.format(single_date))


def multiple_searches(in_file, out_folder, lang='pl', limit=50):
    c = twint.Config()

    with open(in_file, 'r') as f:
        topics = f.read().split(';')

    for topic in topics:
        c.Search = f' {topic} '
        c.Lang = lang
        c.Limit = limit

        c.Store_csv = True
        c.Output = out_folder + f'/{topic.replace(" ", "_")}.csv'

        twint.run.Search(c)
        print(f'Finished for: "{topic}"!')
    print('Finish!')

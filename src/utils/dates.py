from datetime import datetime, timedelta


def daterange(start, end):
    date0 = datetime.strptime(start, "%Y-%m-%d")
    date1 = datetime.strptime(end, "%Y-%m-%d")
    for n in range(int((date1-date0).days)):
        yield date0 + timedelta(n)

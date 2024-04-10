from datetime import datetime, timedelta

def get_next_begin_train_date(input_date_str):
    """
        更新训练集开始时间为 下一年的第一天
    """
    # 将输入日期字符串转换为日期对象
    input_date = datetime.strptime(input_date_str, '%Y-%m-%d')
    
    # 计算下一年的日期
    next_year_date = input_date + timedelta(days=366)
    
    # 计算下一年的第一天日期
    next_year_first_day = datetime(next_year_date.year, 1, 1)
    
    # 将结果日期对象转换为日期字符串
    next_year_first_day_str = next_year_first_day.strftime('%Y-%m-%d')
    
    return next_year_first_day_str


def get_end_train_date(begin_train_date):
    """
        输出训练集结束时间（四年训练集）
    """
    input_date = datetime.strptime(begin_train_date, '%Y-%m-%d')
    
    # 计算两年后的日期
    two_years_later = input_date + timedelta(days=3 * 366)
    
    # 计算两年后日期的最后一天日期
    year = two_years_later.year
    month = 12
    day = 31
    last_day_of_two_years = datetime(year, month, day)
    
    # 将结果日期对象转换为日期字符串
    last_day_of_two_years_str = last_day_of_two_years.strftime('%Y-%m-%d')
    
    return last_day_of_two_years_str


def get_begin_valid_date(end_train_date):
    """
        输出测试集开始时间（一年测试集）
    """
    input_date = datetime.strptime(end_train_date, '%Y-%m-%d')
    next_day_date = input_date + timedelta(days=1)
    next_day_date_str = next_day_date.strftime('%Y-%m-%d')

    return next_day_date_str


def get_end_valid_date(begin_valid_date):
    """
        输出测试集结束时间（一年测试集）
    """
    input_date = datetime.strptime(begin_valid_date, '%Y-%m-%d')
    year = input_date.year
    month = 12
    day = 31
    last_day = datetime(year, month, day)
    last_day_str = last_day.strftime('%Y-%m-%d')

    return last_day_str


def get_begin_test_date(end_valid_date):
    """
        输出测试集开始时间（一年测试集）
    """
    input_date = datetime.strptime(end_valid_date, '%Y-%m-%d')
    next_day_date = input_date + timedelta(days=1)
    next_day_date_str = next_day_date.strftime('%Y-%m-%d')

    return next_day_date_str


def get_end_test_date(begin_test_date):
    """
        输出测试集结束时间（一年测试集）
    """
    input_date = datetime.strptime(begin_test_date, '%Y-%m-%d')
    year = input_date.year
    month = 12
    day = 31
    last_day = datetime(year, month, day)
    last_day_str = last_day.strftime('%Y-%m-%d')

    return last_day_str

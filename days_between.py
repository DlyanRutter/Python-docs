def isLeapYear(year):
  if year % 4 != 0:
      return False
  if year % 100 != 0:
      return True
  if year % 400 != 0:
      return False
  else:
      return True  

leap_months = [31,29,31,30,31,30,31,31,30,31,30,31]
no_leap_months = [31,28,31,30,31,30,31,31,30,31,30,31]

def count_year_days(year):
    index = 0
    year_days_one = 0
    year_days_two = 0
    year_days = 0
    while index < year:
        if isLeapYear(index) == True:
            year_days_one = year_days_one + 366
        elif isLeapYear(index) == False:
            year_days_two = year_days_two + 365
        index = index + 1
        year_days = year_days_one + year_days_two
    return year_days

def count_month_days(year, month):
    if isLeapYear(year) == True:
       if month == 1:
           month_days = 0
       elif month == 2:
            month_days = 31
       elif month == 3:
            month_days = 60
       elif month == 4:
            month_days = 91
       elif month == 5:
            month_days = 121
       elif month == 6:
            month_days = 152
       elif month == 7:
            month_days = 182
       elif month == 8:
            month_days = 213
       elif month == 9:
            month_days = 244
       elif month == 10:
            month_days = 274
       elif month == 11:
            month_days = 305
       elif month == 12:
            month_days = 335
    else:
        if month == 1:
           month_days = 0
        elif month == 2:
            month_days = 31
        elif month == 3:
            month_days = 59
        elif month == 4:
            month_days = 90
        elif month == 5:
            month_days = 120
        elif month == 6:
            month_days = 151
        elif month == 7:
            month_days = 181
        elif month == 8:
            month_days = 212
        elif month == 9:
            month_days = 243
        elif month == 10:
            month_days = 273
        elif month == 11:
            month_days = 304
        elif month == 12:
            month_days = 334
    return month_days

def count_month_days2(year, month):
    month_days = 0
    index = 0
    index_two = 1
    if index_two == month:
        month_days = 0
    elif isLeapYear == True:
        while index_two < month:            
            month_days = month_days + leap_months[index]
            index = index + 1
            index_two = index_two + 1
    elif isLeapYear == False:
        while index_two < month:
            month_days = month_days + no_leap_months[index]
            index = index + 1
            index_two = index_two + 1
    return month_days
            
     

def difference_in_days(year1, month1, day1, year2, month2, day2):
    year_days_one = count_year_days(year1)
    month_days_one = count_month_days(year1, month1)
    days_one = day1
    total_days_one = year_days_one + month_days_one + day1
    year_days_two = count_year_days(year2)
    month_days_two = count_month_days(year2, month2)
    days_two = day2
    total_days_two = year_days_two + month_days_two + days_two 
    all_days_right = total_days_two - total_days_one
    all_days_wrong = total_days_one - total_days_two
    if all_days_right >= 0:
        return all_days_right
    else:
        return all_days_wrong

print isLeapYear(2012)
print difference_in_days(2013, 1, 1, 1999, 12, 31)
print difference_in_days(2012, 9, 30, 2012, 10, 30)
print difference_in_days(2012, 2, 1, 2012, 3, 1)
print difference_in_days(2012, 1, 1, 2012, 2, 28)
print difference_in_days(0, 1, 21, 0, 1, 12)




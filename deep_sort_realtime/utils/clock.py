import os
import logging

from datetime import datetime, timedelta

log_level = logging.DEBUG
default_logger = logging.getLogger('Clock (default logger)')
default_logger.setLevel(log_level)
handler = logging.StreamHandler()
handler.setLevel(log_level)
formatter = logging.Formatter('[%(levelname)s] [%(name)s] %(message)s')
handler.setFormatter(formatter)
default_logger.addHandler(handler)

class Clock(object):
    def __init__(self, logger=None):
        if logger is None:
            self.logger = default_logger
        else:
            self.logger = logger

        tz_offset_hr = float(os.environ.get('TIMEZONE_OFFSET','0.0'))
        self.tz_offset = timedelta(hours=tz_offset_hr)
        self.dateOnly_strformat = '%d-%m' 
        self.dateWithYear_strformat = '%y%m%d' 
        self.datetime_strformat = '%Y-%m-%dT%H-%M-%S-%f' 
        self.register_now()
        self.logger.info('Clock started at {}'.format(self.get_now_SGT_str()))

    def register_now(self):
        self.now = datetime.now()

    def get_now_SGT(self):
        '''
        returns datetime object
        '''
        return self.now + self.tz_offset

    def get_now_SGT_str(self):
        '''
        return date and time string
        '''
        return self.get_now_SGT().strftime(self.datetime_strformat)

    def get_now_SGT_date_str(self):
        '''
        returns date (MM-DD) string only
        '''
        return self.get_now_SGT().strftime(self.dateOnly_strformat)

    def get_now_SGT_date_withyear_str(self):
        '''
        returns date (YYMMDD) string only
        '''
        return self.get_now_SGT().strftime(self.dateWithYear_strformat)

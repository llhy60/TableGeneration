import random
import numpy as np


def load_courp(p, join_c=''):
    courp = []
    with open(p, mode='r', encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip("\n").strip("\r\n")
            courp.append(line)
    courp = join_c.join(courp)
    return courp


class Table:
    def __init__(self,
                 ch_dict_path,
                 en_dict_path,
                 cell_box_type='cell',
                 no_of_rows=14,
                 no_of_cols=14,
                 min_txt_len=2,
                 max_txt_len=7,
                 max_span_row_count=3,
                 max_span_col_count=3,
                 max_span_value=20,
                 color_prob=0,
                 cell_max_width=0,
                 cell_max_height=0):
        assert cell_box_type in [
            'cell', 'text'
        ], "cell_box_type must in ['cell', 'text'],cell: use cell location as cell box; text: use location of text in cell as cell box"
        self.cell_box_type = cell_box_type
        self.no_of_rows = no_of_rows
        self.no_of_cols = no_of_cols
        self.max_txt_len = max_txt_len
        self.min_txt_len = min_txt_len
        self.color_prob = color_prob
        self.cell_max_width = cell_max_width
        self.cell_max_height = cell_max_height
        self.max_span_row_count = max_span_row_count
        self.max_span_col_count = max_span_col_count
        self.max_span_value = max_span_value

        self.dict = ''
        self.ch = load_courp(ch_dict_path, '')
        self.en = load_courp(en_dict_path, '')  # "abcdefghijklmnopqrstuvwxyz"

        self.pre_boder_style = {
            1: {
                'name': 'border',
                'style': {
                    'table': 'border:1px solid black;',
                    'td': 'border:1px solid black;',
                    'th': 'border:1px solid black;'
                }
            },  # 绘制全部边框
            2: {
                'name': 'border_top',
                'style': {
                    'table': 'border-top:1px solid black;',
                    'td': 'border-top:1px solid black;',
                    'th': 'border-top:1px solid black;'
                }
            },  # 绘制上横线
            3: {
                'name': 'border_bottom',
                'style': {
                    'table': 'border-bottom:1px solid black;',
                    'td': 'border-bottom:1px solid black;',
                    'th': 'border-bottom:1px solid black;'
                }
            },  # 绘制下横线
            4: {
                'name': 'head_border_bottom',
                'style': {
                    'th': 'border-bottom: 1px solid black;'
                }
            },  # 绘制 head 下横线
            5: {
                'name': 'no_border',
                'style': ''
            },  # 无边框
            6: {
                'name': 'border_left',
                'style': {
                    'table': 'border-left:1px solid black;',
                    'td': 'border-left:1px solid black;',
                    'th': 'border-left:1px solid black;'
                }
            },  # 绘制左竖线
            7: {
                'name': 'border_right',
                'style': {
                    'table': 'border-right:1px solid black;',
                    'td': 'border-right:1px solid black;',
                    'th': 'border-right:1px solid black;'
                }
            },  # 绘制右竖线
            8: {
                'name': 'border_three',
                'style': {
                    'tr:first-child th': 'border-top: 2px solid black;',
                    'tr:nth-child(3) td': 'border-top: 1px solid black;',
                    'tr:last-child td': 'border-bottom: 2px solid black;'
                }
            },  # 绘制三线表
            9: {
                'name': 'border_one',
                'style': {
                    'tr:nth-child(3) td': 'border-top: 1px solid black;',
                    'td': 'width: 144px;'
                }
            },  # 绘制一线表
            10: {
                'name': 'border_dot',
                'style': {
                    'table': 'border-top: solid 2px black;border-bottom: solid 2px black;',
                    'th': 'border: 1px dotted black;',
                    'td': 'border: 1px dotted black;width: 144px;',
                    'th:first-child, td:first-child': 'border-left: none;',
                    'th:last-child, td:last-child': 'border-right: none;'
                }
            },  # 绘制没有左右边框表且里面为虚线
            11: {
                'name': 'no_border_align',
                'style': {
                    'table': 'border-collapse: separate;border-spacing: 40px 0px;',
                    'th:first-child, td:first-child': 'text-align: left;',
                    'th:last-child, td:last-child': 'text-align: right;'
                }
            }, # 绘制没有框且第一列为左对齐，第二列为右对齐
            12: {
                'name': 'border_align',
                'style': {
                    'table': 'width: 60%;',
                    'th:first-child, td:first-child': 'text-align: left;',
                    'th:last-child, td:last-child': 'text-align: right;'
                }
            }, # 绘制没有框且第一列为左对齐，第二列为右对齐
            13: {
                'name': 'no_border_align_padding',
                'style': {
                    'table': 'border-collapse: separate;border-spacing: 40px 0px;',
                    'th': 'align: center;valign: middle;word-break: break-all;padding-bottom: 20px;',
                    'td': 'align: center;valign: middle;word-break: break-all;',
                    'th:first-child, td:first-child': 'text-align: left;',
                    'th:last-child, td:last-child': 'text-align: right;'
                }
            }, # 绘制没有框且第一列为左对齐，第二列为右对齐，并表头向下padding
            14: {
                'name': 'no_border_align_padding',
                'style': {
                    'table': 'border-collapse: collapse;border-spacing: 0;',
                    'th': 'white-space: nowrap;text-align: center;vertical-align: middle;word-break: break-all;border-right:1.5px solid black;border-left:1.5px solid black;padding: 0px 5px 0px 5px;',
                    'tr:nth-child(2) th': 'border-bottom: 2px solid black;',
                    'td': 'white-space: nowrap;text-align: center;vertical-align: middle;word-break: break-all;border-right:1.5px solid black;border-left:1.5px solid black;padding: 0px 5px 0px 5px;',
                }
            }, # 绘制银行流水账单版式1
        }

        # 随机选择一种
        # self.border_type = random.choice(list(self.pre_boder_style.keys()))
        self.border_type = 14

        self.spanflag = False
        '''cell_types matrix have two possible values:
            c: ch
            e: en
            n: number
            t: ''
            m: money

        '''
        self.cell_types = np.chararray(shape=(self.no_of_rows,
                                              self.no_of_cols))
        '''headers matrix have two possible values: 's' and 'h' where 'h' means header and 's' means simple text'''
        self.headers = np.chararray(shape=(self.no_of_rows, self.no_of_cols))
        '''A positive value at a position in matrix shows the number of columns to span and -1 will show to skip that cell as part of spanned cols'''
        self.col_spans_matrix = np.zeros(shape=(self.no_of_rows,
                                                self.no_of_cols))
        '''A positive value at a position means number of rows to span and -1 will show to skip that cell as part of spanned rows'''
        self.row_spans_matrix = np.zeros(shape=(self.no_of_rows,
                                                self.no_of_cols))
        '''missing_cells will contain a list of (row,column) pairs where each pair would show a cell where no text should be written'''
        self.missing_cells = []

        # header_count will keep track of how many top rows and how many left columns are being considered as headers
        self.header_count = {'r': 2, 'c': 0}

    def get_log_value(self):
        ''' returns log base 2 (x)'''
        import math
        return int(math.log(self.no_of_rows * self.no_of_cols, 2))

    def define_col_types(self):
        '''
        We define the data type that will go in each column. We categorize data in three types:
        1. 'n': Numbers
        2. 'w': word
        3. 'r': other types (containing special characters)

        '''

        prob_words = 0.3
        prob_numbers = 0.5
        prob_ens = 0.1
        prob_money = 0.1
        for i, type in enumerate(
                random.choices(
                    ['n', 'm', 'c', 'e'],
                    weights=[prob_numbers, prob_money, prob_words, prob_ens],
                    k=self.no_of_cols)):
            self.cell_types[:, i] = type
        '''The headers should be of type word'''
        self.cell_types[0:2, :] = 'c'
        '''All cells should have simple text but the headers'''
        self.headers[:] = 's'
        # 采购
        # self.headers[0:1, :] = 'h'
        self.headers[0:2, :] = 'h'

    def generate_random_text(self, type):
        '''cell_types matrix have two possible values:
            c: ch
            e: en
            n: number
            t: ''
            m: money

        '''
        if type in ['n', 'm']:
            max_num = random.choice([10, 100, 1000, 10000, 100000, 1000000])
            if random.random() < 0.5:
                out = '{:.2f}'.format(random.random() * max_num)
            elif random.random() < 0.7:
                out = '{:.0f}'.format(random.random() * max_num)
            else:
                # 随机保留小数点后2位
                out = str(random.random() *
                          max_num)[:len(str(max_num)) + random.randint(0, 3)]
            if type == 'm':
                # out = '$' + out
                out = '{:.2f}'.format(float(out))
        elif (type == 'e'):
            txt_len = random.randint(self.min_txt_len, self.max_txt_len)
            out = self.generate_text(txt_len, self.en)
            # 50% 的概率第一个字母大写
            if random.random() < 0.5:
                out[0] = out[0].upper()
        elif type == 't':
            out = ''
        else:
            txt_len = random.randint(self.min_txt_len, self.max_txt_len)
            out = self.generate_text(txt_len, self.ch)
        return ''.join(out)

    def generate_text(self, txt_len, dict):
        random_star_idx = random.randint(0, len(dict) - txt_len)
        txt = dict[random_star_idx:random_star_idx + txt_len]
        return list(txt)

    def agnostic_span_indices(self, maxvalue, max_num=3):
        '''Spans indices. Can be used for row or col span
        Span indices store the starting indices of row or col spans while span_lengths will store
        the length of span (in terms of cells) starting from start index.'''
        span_indices = []
        span_lengths = []
        # random select span count
        span_count = random.randint(1, max_num)
        if (span_count >= maxvalue):
            return [], []

        indices = sorted(random.sample(list(range(0, maxvalue)), span_count))

        # get span start idx and span value
        starting_index = 0
        for i, index in enumerate(indices):
            if (starting_index > index):
                continue

            # max_lengths = maxvalue - index
            # if (max_lengths < 2):
            #     break
            # len_span = random.randint(1, min(max_lengths, self.max_span_value))
            len_span = 2

            if (len_span > 1):
                span_lengths.append(len_span)
                span_indices.append(index)
                starting_index = index + len_span

        return span_indices, span_lengths

    def make_first_row_spans(self):
        '''This function spans first row'''
        while (True):  # iterate until we get some first row span indices
            header_span_indices, header_span_lengths = self.agnostic_span_indices(
                self.no_of_cols, self.max_span_col_count)
            if (len(header_span_indices) != 0 and
                    len(header_span_lengths) != 0):
                break

        # make first row span matric
        row_span_indices = []
        for index, length in zip(header_span_indices, header_span_lengths):
            self.spanflag = True
            self.col_spans_matrix[0, index] = length
            self.col_spans_matrix[0, index + 1:index + length] = -1
            row_span_indices += list(range(index, index + length))

        # for not span cols, set it to row span value 2
        b = list(
            filter(lambda x: x not in row_span_indices,
                   list(range(self.no_of_cols))))
        self.row_spans_matrix[0, b] = 2
        self.row_spans_matrix[1, b] = -1

    def make_first_col_spans(self):
        '''To make some random row spans on first col of each row'''
        colnumber = 0
        # skip top 2 rows of header
        span_indices, span_lengths = self.agnostic_span_indices(
            self.no_of_rows - 2, self.max_span_row_count)
        span_indices = [x + 2 for x in span_indices]

        for index, length in zip(span_indices, span_lengths):
            if index < 2:
                continue
            self.spanflag = True
            # self.row_spans_matrix[index, colnumber] = length
            # self.row_spans_matrix[index + 1:index + length, colnumber] = -1
            self.row_spans_matrix[index, :5] = length
            self.row_spans_matrix[index, 6:10] = length
            self.row_spans_matrix[index + 1:index + length, :5] = -1
            self.row_spans_matrix[index + 1:index + length, 6:10] = -1
        self.headers[:, colnumber] = 'h'
        self.header_count['c'] += 1

    def generate_missing_cells(self):
        '''This is randomly select some cells to be empty (not containing any text)'''
        missing = np.random.random(size=(self.get_log_value(), 2))
        missing[:, 0] = (self.no_of_rows - 1 - self.header_count['r']
                         ) * missing[:, 0] + self.header_count['r']
        missing[:, 1] = (self.no_of_rows - 1 - self.header_count['c']
                         ) * missing[:, 1] + self.header_count['c']
        for arr in missing:
            self.missing_cells.append((int(arr[0]), int(arr[1])))

    def create_style(self):
        '''This function will dynamically create stylesheet. This stylesheet essentially creates our specific
        border types in tables'''
        boder_style = self.pre_boder_style[self.border_type]['style']
        style = '<head><meta charset="UTF-8"><style>'
        style += "html{background-color: white;}table{"

        # 表格中文本左右对齐方式
        style += "text-align:{};".format(
            random.choices(
                ['left', 'right', 'center'], weights=[0.25, 0.25, 0.5])[0])
        style += "border-collapse:collapse;"
        if 'table' in boder_style:
            style += boder_style['table']
        style += "}td{"

        # 文本上下居中
        if random.random() < 0.5:
            style += "align: center;valign: middle;"
        # 大单元格
        if self.cell_max_height != 0:
            style += "height: {}px;".format(
                random.randint(self.cell_max_height // 2,
                               self.cell_max_height))
        if self.cell_max_width != 0:
            style += "width: {}px;".format(
                random.randint(self.cell_max_width // 2, self.cell_max_width))
        # 文本换行
        style += "word-break:break-all;"
        if 'td' in boder_style:
            style += boder_style.pop('td')

        style += "}th{padding:6px;padding-left: 15px;padding-right: 15px;"
        if 'th' in boder_style:
            style += boder_style.pop('th')
        style += '}'

        for key, value in boder_style.items():
            style += "{}".format(key) + "{" + "{}".format(value) + "}"

        style += "</style></head>"
        return style

    def create_style_my(self):
        '''This function will dynamically create stylesheet. This stylesheet essentially creates our specific
        border types in tables'''
        boder_style = self.pre_boder_style[self.border_type]['style']
        style = '<head><meta charset="UTF-8"><style>'
        style += "html{background-color: white;}table{"
        # style += "html{background-color: rgb(144, 132, 120);}table{"

        # 表格中文本左右对齐方式(其他关联方不使用)
        style += "text-align:{};".format(
            random.choices(
                ['left', 'right', 'center'], weights=[0., 0., 1.])[0])
        style += "border-collapse:collapse;"
        if 'table' in boder_style:
            style += boder_style['table']
        style += "}td{"

        # 文本上下居中
        if random.random() < 0.5:
            style += "align: center;valign: middle;"
        # 大单元格
        if self.cell_max_height != 0:
            style += "height: {}px;".format(
                random.randint(self.cell_max_height // 2,
                               self.cell_max_height))
        if self.cell_max_width != 0:
            style += "width: {}px;".format(
                random.randint(self.cell_max_width // 2, self.cell_max_width))
        # 文本换行
        # style += "word-break:break-all;"
        if 'td' in boder_style:
            style += boder_style['td']

        style += "}th{padding:6px;padding-left: 15px;padding-right: 15px;"
        if 'th' in boder_style:
            style += boder_style['th']
        style += '}'
        if isinstance(boder_style, dict):
            for key, value in boder_style.items():
                style += f"{key}" + "{" + f"{value}" + "}"
        # if 'tr:first-child th' in boder_style:
        #     style += "tr:first-child th {border-top: 2px solid black;}tr:nth-child(3) td {border-top: 1px solid black;}tr:last-child td {border-bottom: 2px solid black;}"
        style += "</style></head>"
        return style

    def create_style_other(self):
        '''This function will dynamically create stylesheet. This stylesheet essentially creates our specific
        border types in tables'''
        boder_style = self.pre_boder_style[self.border_type]['style']
        style = '<head><meta charset="UTF-8"><style>'
        style += "html{background-color: white;}table{"
        # style += "html{background-color: rgb(144, 132, 120);}table{"

        # 表格中文本左右对齐方式(其他关联方不使用)
        style += "border-collapse:collapse;"
        if 'table' in boder_style:
            style += boder_style['table']
        style += "}th,td{"

        # 文本上下居中
        if random.random() < 0.5:
            style += "align: center;valign: middle;"
        # 大单元格
        if self.cell_max_height != 0:
            style += "height: {}px;".format(
                random.randint(self.cell_max_height // 2,
                               self.cell_max_height))
        if self.cell_max_width != 0:
            style += "width: {}px;".format(
                random.randint(self.cell_max_width // 2, self.cell_max_width))
        # 文本换行
        style += "word-break:break-all;}"

        if isinstance(boder_style, dict):
            for key, value in boder_style.items():
                style += f"{key}" + "{" + f"{value}" + "}"
        # if 'tr:first-child th' in boder_style:
        #     style += "tr:first-child th {border-top: 2px solid black;}tr:nth-child(3) td {border-top: 1px solid black;}tr:last-child td {border-bottom: 2px solid black;}"
        style += "</style></head>"
        return style

    def create_html_my(self):
        '''Depending on various conditions e.g. columns spanned, rows spanned, data types of columns,
        regular or irregular headers, tables types and border types, this function creates equivalent html
        script'''
        # 应收版式
        yingshou_col = [[0.,  0.,  2., -1.,  2., -1.], [0.,  0.,  0.,  0.,  0.,  0.]]
        yingshou_row = [[2.,  2.,  0.,  0.,  0.,  0.], [-1., -1.,  0.,  0.,  0.,  0.]]
        init_txt_1 = np.array([['项目名称', '关联方', '年末余额', '', '年初余额', ''],
                             ['', '', '账面余额', '坏账准备', '账面余额', '坏账准备']])
        init_company_1 = ['上海国际港务(集团)股份有限公司', '上海海通国际汽车码头有限公司', '锦江航运(泰国)代理有限公司', '中集世联达物流科技（集团）股份有限公司', '东方海外货柜航运（中国)有限公司', '中远海运物流供应链有限公司', '沧州上港物流有限公司', '上海泛亚航运有限公司', '中远海运集装箱运输有限公司', '港联捷(上海)物流科技有限公司', '上海中远海运物流有限公司', '民生轮船股份有限公司', '上港集团长江物流江西有限公司', '上海同景国际物流发展有限公司', '安徽海润信息技术有限公司', '上港外运集装箱仓储服务有限公司', '上海港海铁联运有限公司', '上海浦海航运有限公司', '上海中远海运集装箱运输有限公司', '上海中远海运集装箱船务代理有限公司', '上海新港集装箱物流有限公司', '上港集团长江物流湖南有限公司', '上海上港瀛东商贸有限公司', '锦茂国际物流（上海)有限公司', '上海海通国际汽车物流有限公司', '新鑫海航运有限公司', '武汉港集装箱有限公司', '江西港铁物流发展有限公司', '上港集团长江物流湖北有限公司', '厦门智图思科技有限公司', '江苏中远海运集装箱运输有限公司', '洋山同盛港口建设有限公司', '安吉上港国际港务有限公司', '万航旅业(上海)有限公司', '广州远海汽车船运输有限公司', '洋山申港国际石油储运有限公司', '中建港航局集团有限公司', '上海锦江三井仓库国际物流有限公司', '湖州上港国际港务有限公司', '上海汉唐航运有限公司', '重庆集海航运有限责任公司', '上海仁川国际渡轮有限公司', '上海奥吉实业有限公司', '江阴苏南国际集装箱码头有限公司', '芜湖港务有限责任公司', '上海浦之星餐饮发展有限公司', '上海尚九一滴水餐饮管理有限公司', '太仓正和国际集装箱码头有限公司', '武汉中远海运集装箱运输有限公司', '上海海通国际汽车码头有限公司', '中石油上港能源有限公司', '东方海外货柜航运（中国)有限公司', '民生轮船股份有限公司', '上海泛亚航运有限公司', '上海亿通国际股份有限公司', '上海中远海运集装箱运输有限公司', '上港集团长江物流湖北有限公司', '中远海运物流供应链有限公司']
        
        # 子公司版式
        zigongsi_col = [[0., 0., 0., 0., 0., 2., -1., 0.], [0., 0., 0., 0., 0., 0., 0., 0.]]
        zigongsi_row = [[2., 2., 2., 2., 2., 0., 0., 2.], [-1., -1., -1., -1., -1, 0., 0., -1.]]
        init_txt_2 = np.array([['子公司名称', '主要经营地', '注册资本 (万元）', '注册地', '业务性质', '持股比例(%)', '', '取得方式'],
                               ['', '', '', '', '', '直接', '间接', '']])
        init_company_2 = ['上海国际港务(集团)股份有限公司', '上海海通国际汽车码头有限公司', '锦江航运(泰国)代理有限公司', '中集世联达物流科技（集团）股份有限公司', '东方海外货柜航运（中国)有限公司', '中远海运物流供应链有限公司', '沧州上港物流有限公司', '上海泛亚航运有限公司', '中远海运集装箱运输有限公司', '港联捷(上海)物流科技有限公司', '上海中远海运物流有限公司', '民生股份有限公司', '长江物流江西有限公司', '上海同景国际物流发展有限公司', '安徽海润信息技术有限公司', '上港外运集装箱仓储服务仓储服务仓储服务有限公司', '上海港海铁联运港海铁联运港海铁联运有限公司', '上海浦海航运上海浦海航运上海浦海航运有限公司', '上海中远海运集装箱运输有限公司', '上海中远海运集装箱船务代理有限公司', '上海新港集装箱物流有限公司', '上港集团长江物流湖南有限公司', '海瀛东商贸有限公司', '锦茂国际物流（上海)有限公司', '上海海通国际汽车物流有限公司', '新鑫海航运有限公司', '武汉港集装箱港集装箱港集装箱有限公司', '江西港铁物流发展港集装箱港集装箱有限公司', '上港集团长江物流湖北有限公司', '厦门智图思科技有限公司', '江苏中远海运集装箱运输有限公司', '洋山同盛港口建设有限公司', '安吉上港国际港务有限公司', '万航旅业(上海)有限公司', '广州远海汽车船运输有限公司', '洋山申港国际石油储运有限公司', '中建港航局集团有限公司', '上海锦江三井仓库国际物流三井仓库国际物流三井仓库国际物流有限公司', '湖州上港国际港务有限公司', '上海汉唐航运有限公司', '重庆集海航运集海航运集海航运有限责任公司', '上海仁川国际渡轮有限公司', '上海奥吉实业有限公司', '江阴苏南国际集装箱码头有限公司', '芜湖港务有限责任公司', '餐饮发展有限公司', '上海尚九一滴水餐饮管理有限公司', '太仓正和国际集装箱码头有限公司', '武汉中远海运集装箱运输有限公司', '上海海通国际汽车码头有限公司', '中石油上港能源有限公司', '东方海外货柜航运有限公司', '民生轮船有限公司', '泛亚航运有限公司', '上海亿通国际股份有限公司', '上海中远海运集装箱运输有限公司', '上港集团长江物流湖北有限公司', '中远海运物流供应链有限公司', '泛亚航运有限公司', '万航旅业有限公司']
        init_city_2 = ['武汉', '天津', '重庆', '西安', '长春', '广州', '深圳', '上海', '成都', '北京']

        # 采购版式
        caigou_col = [[0., 0., 0., 0.], [0., 0., 0., 0.]]
        caigou_row = [[0., 0., 0., 0.], [0., 0., 0., 0.]]
        init_txt_3 = np.array(['关联方', '关联交易内容', '本期发生额', '上期发生额'])
        init_relatation = ['西藏康泽药业发展有限公司', 'NAVAMEDIC ASA'] + init_company_2
        init_content = ['药品推广服务等', '伊姆多产品服务', '采购心脏支架', '物业管理服务、绿化服务', '销售依姆多', '销售大健康产品', '销售大健康产品', '销售大健康产品']

        # 其他关联方
        other_col = [[0., 0.], [0., 0.]]
        other_row = [[2., 2.], [-1., -1.]]
        init_txt_4 = np.array([['公司名称', '与本集团的关系'], ['', '']])
        init_relatation_1 = ['上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之子公司', '上汽总公司之联营企业', '上汽总公司之联营企业', '上汽总公司之联营企业', '上汽总公司之联营企业', '上汽总公司之联营企业', '上汽总公司之联营企业', '上汽总公司之联营企业', '上汽总公司之联营企业', '上汽总公司之联营企业', '上汽总公司之联营企业', '本公司执行董事及其他高级管理人员', '关键管理人员', '受同一最终控制方控制', '受同一最终控制方控制', '受同一最终控制方控制', '受同一最终控制方控制', '受同一最终控制方控制', '受同一最终控制方控制', '受同一最终控制方控制', '受同一最终控制方控制', '受同一最终控制方控制', '受同一最终控制方控制', '受同一最终控制方控制']
        init_company_3 = ['房车生活家（福建）出行服务有限公司', '房车生活家（上海）国际旅行社有限公司', '环球车享（福建)汽车租赁有限公司', '环球车享（济南）汽车租赁有限公司', '环球车享（昆明）汽车租赁有限公司', '环球车享（临沂）汽车租赁有限公司', '环球车享（潍坊）汽车租赁有限公司', '环球车享（烟台）汽车租赁有限公司', '环球车享（忻州）汽车租赁有限公司', '环球车享绵阳汽车租赁有限公司', '环球车享南昌汽车租赁有限公司', '环球车享汽车租赁南通有限公司', '环球车享徐州汽车租赁有限公司', '环球车享诸暨汽车租赁有限公司', '上海爱为途篝汽车租赁服务有限公司', '上海阔步实业有限公司', '上海汽车工业房地产开发有限公司', '上海汽车工业有限公司', '深圳上汽南方汽车销售服务有限公司', '深圳市上汽南方实业有限公司', '浙江丽水驿动新能源汽车运营服务有限公司', '浙江衢州驿动新能源汽车运营服务有限公司', '厦门赛可出行科技服务有限公司', '环球车享悦达盐城汽车租赁有限公司', '上海随申行智慧交通科技有限公司', '上海爱迪特设施管理有限公司', '上海安驰企业经营管理有限公司', '上海车城物业管理有限公司', '上海大众汽车礼品有限公司', '上海国际汽车城发展有限公司', '上海汽车博物馆有限公司', '上海上汽索迪斯服务有限公司', '智能汽车创新发展平台（上海）有限公司', '上海机械工业内燃机检测所有限责任公司']

        self.col_spans_matrix[0:2, :] = other_col
        self.row_spans_matrix[0:2, :] = other_row
        # 子公司版式
        # self.row_spans_matrix[:, 7][2] = len(self.row_spans_matrix) - 2
        # self.row_spans_matrix[:, 7][3:] = -1
        idcounter = 0
        structure = []
        temparr = ['td', 'th']
        html = """<html>"""
        html += self.create_style_other()
        html += '<body><table>'
        # html += '<table style="width: 100%; table-layout:fixed;">'
        for r in range(self.no_of_rows):
            html += '<tr>'
            structure.append('<tr>')
            for c in range(self.no_of_cols):
                text_type = self.cell_types[r, c].decode('utf-8')
                row_span_value = int(self.row_spans_matrix[r, c])
                col_span_value = int(self.col_spans_matrix[r, c])
                htmlcol = temparr[['s', 'h'].index(self.headers[r][c].decode(
                    'utf-8'))]
                if self.cell_box_type == 'cell':
                    htmlcol += ' id={}'.format(idcounter)
                htmlcol_style = htmlcol
                # set color
                if (col_span_value != 0) or (r, c) not in self.missing_cells:
                    if random.random() < self.color_prob:
                        color = (random.randint(200, 255), random.randint(
                            200, 255), random.randint(200, 255))
                        htmlcol_style += ' style="background-color: rgba({}, {}, {},1);"'.format(
                            color[0], color[1], color[2])

                if (row_span_value == -1):
                    continue

                elif (row_span_value > 0):
                    html += '<' + htmlcol_style + ' rowspan=\"' + str(
                        row_span_value) + '">'
                    if row_span_value > 1:
                        structure.append('<td')
                        structure.append(' rowspan=\"{}\"'.format(
                            row_span_value))
                        structure.append('>')
                    else:
                        structure.append('<td>')
                else:
                    if (col_span_value == 0):
                        if (r, c) in self.missing_cells:
                            text_type = 't'
                    if (col_span_value == -1):
                        continue
                    html += '<' + htmlcol_style + ' colspan=\"' + str(
                        col_span_value) + '">'
                    if col_span_value > 1:
                        structure.append('<td')
                        structure.append(' colspan=\"{}\"'.format(
                            col_span_value))
                        structure.append('>')
                    else:
                        structure.append('<td>')
                if c == 0:
                    # 第一行设置为中英文
                    text_type = random.choice(['c', 'e'])
                txt = self.generate_random_text(text_type)

                # 采购
                # if r == 0:
                #     txt = init_txt_3[c]
                # elif r > 0 and c == 0:
                #     txt = random.choice(init_relatation)
                # elif r > 0 and c == 1:
                #     txt = random.choice(init_content)
                # elif r > 0 and c >= 2:
                #     txt = self.generate_random_text('m')

                # 其它关联方
                if r == 0:
                    txt = init_txt_4[r][c]
                    if not txt:
                        continue
                elif r > 0 and c == 0:
                    txt = random.choice(init_company_3)
                elif r > 0 and c == 1:
                    txt = random.choice(init_relatation_1)

                # 子公司版式
                # if r == 0:
                #     txt = init_txt_5[c]
                # elif r > 0 and c == 0:

                # 子公司版式
                # if 0 <= r < 2:
                #     txt = init_txt_2[r][c]
                #     if not txt:
                #         continue
                # elif r >= 2 and c == 0:
                #     txt = random.choice(init_company_2)
                # elif r >= 2 and c == 1:
                #     txt = random.choice(init_city_2)
                # elif r >= 2 and c == 2:
                #     txt = self.generate_random_text('m')
                # elif r >= 2 and c == 3:
                #     txt = random.choice(init_city_2)
                # elif r >= 2 and c == 4:
                #     txt = '勘察设计'
                # elif r >= 2 and c == 5:
                #     txt = '100.00'
                # elif r >= 2 and c == 6:
                #     txt = ' '
                # elif r >= 2 and c == 7:
                #     txt = '见"注"'

                # if r >= 2 and c == 0 and self.cell_box_type == 'text':
                #     txt = '<span id=' + str(idcounter) + '>' + txt + ' </span>'
                idcounter += 1
                html += txt + '</' + htmlcol + '>'
                structure.append('</td>')

            html += '</tr>'
            structure.append('</tr>')
        html += "<table></body></html>"
        return html, structure, idcounter

    def create_html_type1(self):
        '''Depending on various conditions e.g. columns spanned, rows spanned, data types of columns,
        regular or irregular headers, tables types and border types, this function creates equivalent html
        script'''
        # import pdb; pdb.set_trace()
        HEADERS = np.array([
            ['序号', '记账日', '起息日', '交易类型', '凭证', '凭证号码/业务编号/用途/摘要', '借方发生额', '贷方发生额', '余额', '机构/柜员/流水', '备注'],
            ['No.', 'Bk. D.', 'Val. D.', 'Type', 'Vou.', 'Vou.No./Trans.No./Details', 'Debit Amount', 'Credit Amount', 'Balance', 'Reference No.', 'Notes']])
        idcounter = 0
        structure = []
        temparr = ['td', 'th']
        html = """<html>"""
        html += self.create_style()
        html += '<body><table>'
        # html += '<table style="width: 100%; table-layout:fixed;">'
        for r in range(self.no_of_rows):
            html += '<tr>'
            structure.append('<tr>')
            for c in range(self.no_of_cols):
                text_type = self.cell_types[r, c].decode('utf-8')
                row_span_value = int(self.row_spans_matrix[r, c])
                col_span_value = int(self.col_spans_matrix[r, c])
                htmlcol = temparr[['s', 'h'].index(self.headers[r][c].decode(
                    'utf-8'))]
                if self.cell_box_type == 'cell':
                    htmlcol += ' id={}'.format(idcounter)
                htmlcol_style = htmlcol
                # set color
                if (col_span_value != 0) or (r, c) not in self.missing_cells:
                    if random.random() < self.color_prob:
                        color = (random.randint(200, 255), random.randint(
                            200, 255), random.randint(200, 255))
                        htmlcol_style += ' style="background-color: rgba({}, {}, {},1);"'.format(
                            color[0], color[1], color[2])

                if (row_span_value == -1):
                    continue

                elif (row_span_value > 0):
                    html += '<' + htmlcol_style + ' rowspan=\"' + str(
                        row_span_value) + '">'
                    if row_span_value > 1:
                        structure.append('<td')
                        structure.append(' rowspan=\"{}\"'.format(
                            row_span_value))
                        structure.append('>')
                    else:
                        structure.append('<td>')
                else:
                    if (col_span_value == 0):
                        if (r, c) in self.missing_cells:
                            text_type = 't'
                    if (col_span_value == -1):
                        continue
                    html += '<' + htmlcol_style + ' colspan=\"' + str(
                        col_span_value) + '">'
                    if col_span_value > 1:
                        structure.append('<td')
                        structure.append(' colspan=\"{}\"'.format(
                            col_span_value))
                        structure.append('>')
                    else:
                        structure.append('<td>')
                if r < 2:
                    txt = HEADERS[r][c]
                else:
                    if c == 0:
                        txt = str(r - 1)
                    elif 0 < c < 3:
                        txt = '220901'
                    elif c == 3:
                        txt = random.choices(['收费', '小额普通'], weights=[0.5, 0.5])[0]
                    elif c == 4:
                        txt = ''
                    elif c == 5:
                        txt = random.choices(['转账汇款手续费（网银）', 'BEPS1042900851832022090202288904/广汉办外协', '车运费/广汉办外协车运费'], weights=[0.3, 0.4, 0.3])[0]
                    elif c == 6:
                        txt = random.choices(['', self.generate_random_text("m")])[0]
                        flag = 1 if txt else 0
                    elif c == 7 and flag:
                        txt = ''
                    elif c == 8:
                        txt = self.generate_random_text("m")
                    elif c == 9:
                        txt = random.choices(['05856/9880100/156402289', '05856/9880100/154308012'])[0]
                    elif c == 10:
                        txt = random.choices(['', '刘登模/中国农业银行股份有', '限公司定远县支行'])[0]
                    else:
                        txt = self.generate_random_text("m")
                if self.cell_box_type == 'text':
                    txt = '<span id=' + str(idcounter) + '>' + txt + ' </span>'
                idcounter += 1
                html += txt + '</' + htmlcol + '>'
                structure.append('</td>')

            html += '</tr>'
            structure.append('</tr>')
        html += "<table></body></html>"
        return html, structure, idcounter

    def create_html(self):
        '''Depending on various conditions e.g. columns spanned, rows spanned, data types of columns,
        regular or irregular headers, tables types and border types, this function creates equivalent html
        script'''
        idcounter = 0
        structure = []
        temparr = ['td', 'th']
        html = """<html>"""
        html += self.create_style()
        html += '<body><table>'
        # html += '<table style="width: 100%; table-layout:fixed;">'
        for r in range(self.no_of_rows):
            html += '<tr>'
            structure.append('<tr>')
            for c in range(self.no_of_cols):
                text_type = self.cell_types[r, c].decode('utf-8')
                row_span_value = int(self.row_spans_matrix[r, c])
                col_span_value = int(self.col_spans_matrix[r, c])
                htmlcol = temparr[['s', 'h'].index(self.headers[r][c].decode(
                    'utf-8'))]
                if self.cell_box_type == 'cell':
                    htmlcol += ' id={}'.format(idcounter)
                htmlcol_style = htmlcol
                # set color
                if (col_span_value != 0) or (r, c) not in self.missing_cells:
                    if random.random() < self.color_prob:
                        color = (random.randint(200, 255), random.randint(
                            200, 255), random.randint(200, 255))
                        htmlcol_style += ' style="background-color: rgba({}, {}, {},1);"'.format(
                            color[0], color[1], color[2])

                if (row_span_value == -1):
                    continue

                elif (row_span_value > 0):
                    html += '<' + htmlcol_style + ' rowspan=\"' + str(
                        row_span_value) + '">'
                    if row_span_value > 1:
                        structure.append('<td')
                        structure.append(' rowspan=\"{}\"'.format(
                            row_span_value))
                        structure.append('>')
                    else:
                        structure.append('<td>')
                else:
                    if (col_span_value == 0):
                        if (r, c) in self.missing_cells:
                            text_type = 't'
                    if (col_span_value == -1):
                        continue
                    html += '<' + htmlcol_style + ' colspan=\"' + str(
                        col_span_value) + '">'
                    if col_span_value > 1:
                        structure.append('<td')
                        structure.append(' colspan=\"{}\"'.format(
                            col_span_value))
                        structure.append('>')
                    else:
                        structure.append('<td>')
                if c == 0:
                    # 第一行设置为中英文
                    text_type = random.choice(['c', 'e'])
                txt = self.generate_random_text(text_type)
                if self.cell_box_type == 'text':
                    txt = '<span id=' + str(idcounter) + '>' + txt + ' </span>'
                idcounter += 1
                html += txt + '</' + htmlcol + '>'
                structure.append('</td>')

            html += '</tr>'
            structure.append('</tr>')
        html += "<table></body></html>"
        return html, structure, idcounter

    def create(self):
        '''This will create the complete table'''
        self.define_col_types()  # define the data types for each column
        self.generate_missing_cells()  # generate missing cells

        if self.border_type < 60:  # 绘制横线的情况下进行随机span
            # first row span
            # if self.max_span_col_count > 0:
            #     self.make_first_row_spans()
            # first col span
            if random.random() < 1 and self.max_span_row_count > 0:
                # import pdb; pdb.set_trace()
                self.make_first_col_spans()
        html, structure, idcounter = self.create_html_type1()  # create equivalent html

        return idcounter, html, structure, self.pre_boder_style[
            self.border_type]['name']

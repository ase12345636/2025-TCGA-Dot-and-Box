from PyQt5.QtWidgets import QMainWindow, QApplication, QComboBox, QSlider, QLineEdit
from PyQt5 import QtWidgets
from PyQt5.QtGui import QPainter, QColor, QPen ,QFont,QPixmap
from PyQt5.QtCore import Qt, QTimer
from GUI_style import *
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from Dots_and_Box import *
from RandomBot import Random_Bot,Greedy_Bot
from DeepLearning import *
from Alpha.MCTS import MCTSPlayer
from Alpha.AlphaBeta import AlphaBetaPlayer
from arg import *

args_Res['train'] = False
args_Res['load_model_name'] = None



class GameWindow(QMainWindow):
    def __init__(self):
        super(GameWindow, self).__init__()
        
        self.setStyleSheet(mainWindow_styleSheet)
        
        self.setGeometry(500, 200, 950, 650)
        self.setWindowTitle('Dots and Boxes')
        self.gap = 50
        self.Dots_radius = 12
        self.BoardStart_pos = (60, 60)
        self.mouse_events_enabled = False
        self.paint_events_enabled = False
        self.game_row = 4
        self.game_col = 4
        
        # 設置字體樣式
        self.font = QFont("Arial", 11, QFont.Bold)  # 使用 Arial 字型，大小 14，粗體
        
        
        self.box1_pixmap = QPixmap("GUI_Play/image/box1.png")
        self.box2_pixmap = QPixmap("GUI_Play/image/box2.png")
        
        self.init_ui()

    def init_ui(self):
        
        #玩家1分數
        self.P1_score_label = QtWidgets.QLabel(self)
        self.P1_score_label.setText(f'Player1 scores: 0')
        self.P1_score_label.setGeometry(650, 100, 175, 25)
        self.P1_score_label.setFont(self.font)  # 設置標籤字型

        #玩家2分數
        self.P2_score_label = QtWidgets.QLabel(self)
        self.P2_score_label.setText(f'Player2 scores: 0')
        self.P2_score_label.setGeometry(650, 150, 175, 25)
        self.P2_score_label.setFont(self.font)  # 設置標籤字型

        player_list = ["人類","Random","貪婪","MCTS","AlphaBeta","Resnet"]
        # 玩家1下拉選單
        self.p1_combo_box = QComboBox(self)
        self.p2_combo_box = QComboBox(self)
        for player in player_list:
            self.p1_combo_box.addItem(player)
            self.p2_combo_box.addItem(player)
        self.p1_combo_box.setGeometry(650, 200, 75, 25)
        self.p1_combo_box.setStyleSheet(combo_styleSheet)
        self.p1_combo_box.setFont(self.font)  # 設置下拉選單字型
        
        # 創建ver輸入框
        self.p1_ver_input_box = QLineEdit(self)
        self.p1_ver_input_box.setGeometry(650, 25, 75, 25)
        self.p1_ver_input_box.setPlaceholderText("p1 version")
        
        self.p2_ver_input_box = QLineEdit(self)
        self.p2_ver_input_box.setGeometry(750, 25, 75, 25)
        self.p2_ver_input_box.setPlaceholderText("p2 version")
        
        self.vs_label = QtWidgets.QLabel(self)
        self.vs_label.setText("VS")
        self.vs_label.setGeometry(750, 200, 25, 25)
        self.vs_label.setFont(self.font)
        
        # 玩家2下拉選單
        self.p2_combo_box.setGeometry(775, 200, 75, 25)
        self.p2_combo_box.setStyleSheet(combo_styleSheet)
        self.p2_combo_box.setFont(self.font)  # 設置下拉選單字型
        #調整遊戲行數之滑條
        self.row_slider = QSlider(Qt.Horizontal, self)
        self.row_slider.setGeometry(650,250,150,25)
        self.row_slider.setStyleSheet(slider_styleSheet)
        self.row_slider.setMinimum(3)  # 設定滑桿最小值
        self.row_slider.setMaximum(6)  # 設定滑桿最大值
        self.row_slider.setValue(4)  # 設定滑桿初始值
        self.row_slider.setTickPosition(QSlider.TicksBelow)  # 設置刻度線的位置
        self.row_slider.setTickInterval(1)  # 設置刻度線的間距
        self.row_slider.valueChanged.connect(self.OnRowSlide)  # 當滑桿的值變化時觸發
        
        #調整遊戲列數之滑條
        self.col_slider = QSlider(Qt.Horizontal, self)
        self.col_slider.setGeometry(650,300,150,25)
        self.col_slider.setStyleSheet(slider_styleSheet)
        self.col_slider.setMinimum(3)  # 設定滑桿最小值
        self.col_slider.setMaximum(6)  # 設定滑桿最大值
        self.col_slider.setValue(4)  # 設定滑桿初始值
        self.col_slider.setTickPosition(QSlider.TicksBelow)  # 設置刻度線的位置
        self.col_slider.setTickInterval(1)  # 設置刻度線的間距
        self.col_slider.valueChanged.connect(self.OnColSlide)  # 當滑桿的值變化時觸發
        
        #展示遊戲大小文字
        self.size_label = QtWidgets.QLabel(self)
        self.size_label.setText(f"{self.col_slider.value()} X {self.col_slider.value()}")
        self.size_label.setGeometry(825, 275, 90, 25)
        self.size_label.setFont(self.font)  # 設置下拉選單字型

        #開始按鈕
        self.StartButton = QtWidgets.QPushButton(self)
        self.StartButton.setText("Start!!!")
        self.StartButton.setGeometry(650, 350, 175, 25)
        self.StartButton.setStyleSheet(button_styleSheet)
        self.StartButton.setFont(self.font)  # 設置按鈕字型
        self.StartButton.clicked.connect(self.OnClickStartButton)
        
        #終局文字
        self.winner_label = QtWidgets.QLabel(self)
        self.winner_label.setText("")
        self.winner_label.setGeometry(650,400,200,50)
        self.winner_label.setFont(QFont("Arial", 20, QFont.Bold))
    
    def loadBOT(self, modelBase:str, m, n, ver = None):
        bot = None

        if modelBase == "Resnet":
            if ver:
                args_Res['load_model_name'] = f'Resnet_model_{m}x{n}_{ver}.h5'
            else:
                args_Res['load_model_name'] = None
            bot = ResnetBOT(self.game.state.m, self.game.state.n, self.game, args_Res)

        return bot


    def OnRowSlide(self):
        self.game_row = self.row_slider.value()
        self.size_label.setText(f"{self.game_row} X {self.game_col}")
    def OnColSlide(self):
        self.game_col = self.col_slider.value()
        self.size_label.setText(f"{self.game_row} X {self.game_col}")


    def OnClickStartButton(self):
        #初始化遊戲
        game_state = STATE(
            p1_p2_scores=[0, 0],
            board=[[]],
            m=self.game_row,
            n=self.game_col,
            current_player=-1
        )

        self.game = DotsAndBox(game_state)

        self.P1_score_label.setText(f'Player1 scores: {self.game.state.p1_p2_scores[0]}')
        self.P2_score_label.setText(f'Player2 scores: {self.game.state.p1_p2_scores[1]}')
        self.winner_label.setText("")
        self.StartButton.setText("Reset")
        self.update()
        self.paint_events_enabled = True

        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()  # 停止已經存在的計時器
        
        self.p1 = self.p1_combo_box.currentText()
        if self.p1 == "人類":
            self.p1 = -1
        elif self.p1 == "Random":
            self.p1 = Random_Bot(self.game.state.m, self.game.state.n)
        elif self.p1 == "貪婪":
            self.p1 = Greedy_Bot(self.game.state.m, self.game.state.n)
        elif self.p1 == "MCTS":
            self.p1 = MCTSPlayer(num_simulations=1000, exploration_weight=1.3, max_depth=100, selfFirst=True)
            self.p1.game_state = self.game
        elif self.p1 == "AlphaBeta":
            self.p1 = AlphaBetaPlayer(-1,self.game.state)
        else:
            ver = self.p1_ver_input_box.text()
            self.botOppo = self.loadBOT(modelBase=self.p1,m = self.game.state.m, n = self.game.state.n, ver=ver)
            self.p1 = self.botOppo
        
        self.p2 = self.p2_combo_box.currentText()
        if self.p2 == "人類":
            self.p2 = 1
        elif self.p2 == "Random":
            self.p2 = Random_Bot(self.game.state.m, self.game.state.n)
        elif self.p2 == "貪婪":
            self.p2 = Greedy_Bot(self.game.state.m, self.game.state.n)
        elif self.p2 == "MCTS":
            self.p2 =MCTSPlayer(num_simulations=1000, exploration_weight=1.3, max_depth=100, selfFirst=False)
            self.p2.game_state = self.game
        elif self.p2 == "AlphaBeta":
            self.p2 = AlphaBetaPlayer(1,self.game.state)
        else:
            ver = self.p2_ver_input_box.text()
            self.botOppo = self.loadBOT(modelBase=self.p2,m = self.game.state.m, n = self.game.state.n, ver=ver)
            self.p2 = self.botOppo
      
        # 每過100毫秒檢查一次遊戲並跳至game_loop更新畫面
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.game_loop)
        self.timer.start(100)

    def game_loop(self):
        if isGameOver(self.game.state.board):
            winner = GetWinner(self.game.state.board, self.game.state.p1_p2_scores)
            if winner == -1:
                self.winner_label.setText("Player 1")
                self.winner_label.setStyleSheet("color: #0000E3;")
            elif winner == 0:
                self.winner_label.setText("Tie")
                self.winner_label.setStyleSheet("color: #000000;")
            elif winner == 1:
                self.winner_label.setText("Player 2")
                self.winner_label.setStyleSheet("color: #FF0000;")
            self.StartButton.setText("Start!!!")
            self.timer.stop()
            return

        if self.p1 == -1 and self.p2 != 1:   # p1人類, p2機器
            if self.game.state.current_player == -1:
                self.mouse_events_enabled = True    #人類方使用滑鼠控制棋盤
            else:
                r, c = self.p2.get_move(self.game.state.board, self.game.state.current_player)[0]
                self.game.state.current_player,score = make_move(self.game.state.board, r, c, self.game.state.current_player)
                if self.game.state.current_player == -1:
                    self.game.state.p1_p2_scores[0] += score
                elif self.game.state.current_player == 1:
                    self.game.state.p1_p2_scores[1] += score
                self.update()
                self.mouse_events_enabled = False
        elif self.p1 != -1 and self.p2 == 1:    # p1機器, p2人類
            if self.game.state.current_player == 1:
                self.mouse_events_enabled = True
            else:
                r, c = self.p1.get_move(self.game.state.board, self.game.state.current_player)[0]
                self.game.state.current_player,score = make_move(self.game.state.board, r, c, self.game.state.current_player)
                if self.game.state.current_player == -1:
                    self.game.state.p1_p2_scores[0] += score
                elif self.game.state.current_player == 1:
                    self.game.state.p1_p2_scores[1] += score
                self.update()
                self.mouse_events_enabled = False

        elif self.p1 != -1 and self.p2 != 1:    # p1, p2皆為機器
            self.mouse_events_enabled = False
            if self.game.state.current_player == -1:
                r, c = self.p1.get_move(self.game.state.board, self.game.state.current_player)[0]
            else:
                r, c = self.p2.get_move(self.game.state.board, self.game.state.current_player)[0]
            self.game.state.current_player,score = make_move(self.game.state.board, r, c, self.game.state.current_player)
            if self.game.state.current_player == -1:
                self.game.state.p1_p2_scores[0] += score
            elif self.game.state.current_player == 1:
                self.game.state.p1_p2_scores[1] += score
            self.update()
        else:   #p1, p2皆為人類
            self.mouse_events_enabled = True
    
    #繪畫事件，用於觸發棋盤繪製    
    def paintEvent(self, event):
        if not self.paint_events_enabled:
            return
        painter = QPainter(self)
        self.draw_board(painter)
    
    # 用於繪製遊戲棋盤的方法
    def draw_board(self, painter):
        max_edge = max(self.game.state.m,self.game.state.n)  
        self.gap = 240//(max_edge-1)
        
        self.P1_score_label.setText(f'Player1 scores: {self.game.state.p1_p2_scores[0]}')
        self.P2_score_label.setText(f'Player2 scores: {self.game.state.p1_p2_scores[1]}')
        
        Dots_pen = QPen(QColor('#000000'), 3)
        Blue_solid_pen = QPen(QColor('#00E3E3'), 5)  # 藍色實線，寬度為5
        Red_solid_pen = QPen(QColor('#FF9224'), 5)  # 紅色實線，寬度為5
        dash_pen = QPen(Qt.black, 3)  # 黑色線條，寬度為 3
        dash_pen.setStyle(Qt.CustomDashLine)  # 設置為自定義虛線
        dash_pen.setDashPattern([1, 3, 1, 3])  # 自定義虛線的模式：線段長度 1，空格 3，線段 1，空格 3

        green_pen = QPen(QColor('#00EC00'), 1)

        for i in range(self.game.state.board_rows):
            for j in range(self.game.state.board_cols):
                x = self.BoardStart_pos[0] + j * self.gap
                y = self.BoardStart_pos[1] + i * self.gap
                painter.setPen(green_pen)

                # Debug lines
                # painter.setBrush(Qt.NoBrush)
                # painter.drawRect(x-self.gap//2,y-self.gap//2,self.gap,self.gap)

                painter.setPen(Dots_pen)
                if self.game.state.board[i][j] == -1:
                    painter.setPen(Blue_solid_pen)
                    if i % 2 == 0:  # 偶數列，水平線
                        painter.drawLine(x - self.gap + self.Dots_radius - 1, y, x + self.gap - self.Dots_radius, y)
                    else:
                        painter.drawLine(x, y - self.gap + self.Dots_radius, x, y + self.gap - self.Dots_radius)
                elif self.game.state.board[i][j] == 1:
                    painter.setPen(Red_solid_pen)
                    if i % 2 == 0:  # 偶數列，水平線
                        painter.drawLine(x - self.gap + self.Dots_radius - 1, y, x + self.gap - self.Dots_radius, y)
                    else:
                        painter.drawLine(x, y - self.gap + self.Dots_radius, x, y + self.gap - self.Dots_radius)
                elif self.game.state.board[i][j] == 0:
                    painter.setPen(dash_pen)  # 畫虛線(未選擇的邊)
                    if i % 2 == 0:  # 偶數列，水平線
                        painter.drawLine(x - self.gap + self.Dots_radius - 1, y, x + self.gap - self.Dots_radius, y)
                    else:
                        painter.drawLine(x, y - self.gap + self.Dots_radius, x, y + self.gap - self.Dots_radius)
                elif self.game.state.board[i][j] == 7:
                    painter.drawPixmap(x - self.gap // 2, y - self.gap // 2, self.gap, self.gap, self.box1_pixmap)
                elif self.game.state.board[i][j] == 9:
                   painter.drawPixmap(x - self.gap // 2, y - self.gap // 2, self.gap, self.gap, self.box2_pixmap)
                if self.game.state.board[i][j] == 5:
                    painter.setBrush(QColor('#000000'))  # 設定填充顏色為黑色
                    painter.drawEllipse(x - self.Dots_radius // 2, y - self.Dots_radius // 2, self.Dots_radius, self.Dots_radius)  # 繪製實心圓
                    
    def mousePressEvent(self, event):
        if not self.mouse_events_enabled or (self.p1!=-1 and self.p2!=1):
            return  # 如果滑鼠事件未啟用，則直接返回
        if event.button() == Qt.LeftButton:  # 確保是左鍵點擊
            x = event.x() - self.BoardStart_pos[0] + self.gap//2
            y = event.y() - self.BoardStart_pos[1] + self.gap//2
            
            # 計算點擊的行和列
            row = y // self.gap
            col = x // self.gap
            self.game.state.current_player,score = make_move(self.game.state.board, row, col, self.game.state.current_player)
            if self.game.state.current_player == -1:
                self.game.state.p1_p2_scores[0] += score
            elif self.game.state.current_player == 1:
                self.game.state.p1_p2_scores[1] += score
            self.update()
            if self.game.state.current_player != self.p1 or self.game.state.current_player != self.p2: 
                self.mouse_events_enabled = False
            
              
    
def window():
    app = QApplication(sys.argv)  # 設置系統與app的參數連結
    win = GameWindow()  # 創建視窗物件
    win.show()  # 開啟視窗
    sys.exit(app.exec_())  # 確保視窗關閉是由按下視窗關閉建觸發

window()
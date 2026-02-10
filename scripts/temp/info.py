from manim import *

class TwoCarsPass(Scene):
    def construct(self):
        # 道路（画成长灰色矩形）
        road = Rectangle(width=14, height=3, fill_color=GRAY, fill_opacity=0.5).shift(DOWN*1)
        self.play(FadeIn(road))

        # 车道分隔线
        dashed_line = DashedLine(LEFT*7, RIGHT*7, color=WHITE).shift(DOWN*1)
        self.play(Create(dashed_line))

        # 两辆车：蓝车在上车道向右，红车在下车道向左
        car_blue = Rectangle(width=1.2, height=0.7, fill_color=BLUE, fill_opacity=1).move_to(LEFT*6 + DOWN*0.5)
        label_blue = Text("车 A", font_size=24, color=WHITE).next_to(car_blue, UP, buff=0.2)

        car_red = Rectangle(width=1.2, height=0.7, fill_color=RED, fill_opacity=1).move_to(RIGHT*6 + DOWN*1.5)
        label_red = Text("车 B", font_size=24, color=WHITE).next_to(car_red, DOWN, buff=0.2)

        self.play(FadeIn(car_blue), FadeIn(car_red), FadeIn(label_blue), FadeIn(label_red))

        # 动画：两车驶向中间（相遇点附近）
        self.play(
            car_blue.animate.move_to(DOWN*0.5 + LEFT*0.5),
            car_red.animate.move_to(DOWN*1.5 + RIGHT*0.5),
            run_time=3,
            rate_func=linear
        )

        # 相遇时信息交换（用双向箭头+文字表示）
        arrow = DoubleArrow(
            start=car_blue.get_bottom(),
            end=car_red.get_top(),
            buff=0.1,
            color=YELLOW
        )
        info_text = Text("信息素交换", font_size=28, color=YELLOW).next_to(arrow, RIGHT, buff=0.2)

        self.play(GrowArrow(arrow), Write(info_text))
        self.wait(1.5)

        # 信息消失
        self.play(FadeOut(arrow), FadeOut(info_text))

        # 两车继续驶离，开出画面
        self.play(
            car_blue.animate.move_to(RIGHT*7 + DOWN*0.5),
            car_red.animate.move_to(LEFT*7 + DOWN*1.5),
            run_time=3,
            rate_func=linear
        )

        self.wait(1)
        self.play(FadeOut(car_blue), FadeOut(car_red), FadeOut(label_blue), FadeOut(label_red), FadeOut(road), FadeOut(dashed_line))
        
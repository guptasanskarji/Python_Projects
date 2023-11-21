from turtle import *
color('red', 'grey')
pensize(6)
begin_fill()
while True:
    forward(300)
    left(250)
    if abs(pos()) < 1:
        break
end_fill()
done()
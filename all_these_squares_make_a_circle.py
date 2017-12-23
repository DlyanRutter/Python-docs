import turtle

window = turtle.Screen()
window.bgcolor("blue")
irwin = turtle.Turtle()
irwin.shape("turtle")
irwin.speed(3)
i = 0
x = 0
def make_square():
    i = 0          
    while i < 4:
        irwin.forward(100)
        irwin.right(90)
        i = i + 1

def repeat_square():
    x = 0
    while x < 100:
        make_square()
        irwin.right(10)
        x = x + 1
    

#    mandy = turtle.Turtle()
#    mandy.shape("turtle")
#    mandy.color("yellow")
#    mandy.circle(100)

#    billy = turtle.Turtle()
#    billy.shape("turtle")
#    billy.color("red")
#    x = 0
#    while x < 3:
#        billy.forward(100)
#        billy.right(120)
#        x = x + 1
#    window.exitonclick()
#turtle_shapes()
repeat_square()

"""epuck_controller controller."""

from controller import Robot, Motor

import hand_detection as h_d

# create the Robot instance.
robot = Robot()

TIME_STEP = 64
MAX_SPEED = 6.28


leftMotor = robot.getDevice('left wheel motor')
rightMotor = robot.getDevice('right wheel motor')
leftMotor.setPosition(float('inf'))
rightMotor.setPosition(float('inf'))

leftMotor.setVelocity(0.0)
rightMotor.setVelocity(0.0)

while robot.step(TIME_STEP) != -1:

    if h_d.global_forward:
        leftMotor.setVelocity(0.2 * MAX_SPEED)
        rightMotor.setVelocity(0.2 * MAX_SPEED)
        
    elif h_d.global_backward:
        leftMotor.setVelocity(-0.2 * MAX_SPEED)
        rightMotor.setVelocity(-0.2 * MAX_SPEED)
    
    pass


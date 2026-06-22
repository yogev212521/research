(define (problem hanoi4)
  (:domain hanoi)
  
  (:objects
    d1 d2 d3 d4 - disc
    peg1 peg2 peg3 - peg
  )
  
  (:init
    (on d1 peg1)
    (on d2 peg1)
    (on d3 peg1)
    (on d4 peg1)
    (clear peg2)
    (clear peg3)
  )
  
  (:goal
    (and
      (on d1 peg3)
      (on d2 peg3)
      (on d3 peg3)
      (on d4 peg3)
    )
  )
)

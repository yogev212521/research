(define (problem blocksworld-problem-1)
  (:domain blocksworld)
  (:objects
    a b c - block
  )
  (:init
    (ontable a)
    (ontable b)
    (ontable c)
    (clear a)
    (clear b)
    (clear c)
    (handempty)
    (tests)
  )
  (:goal
    (and
      (on a b)
      (on b c)
    )
  )
)

package utility

import org.scalatest.funsuite.AnyFunSuite
import org.mockito.MockitoSugar


class SlidingWindowUtilTest extends AnyFunSuite with MockitoSugar {

  test("generateInputTargetPairs should generate correct input-target pairs") {
    val tokenIds = Array(1, 2, 3, 4, 5, 6, 7, 8, 9)
    val windowSize = 3
    val stride = 1

    // Expected pairs: (sliding input window, target window)
    val expectedPairs = Seq(
      (Array(1, 2, 3), Array(2, 3, 4)),
      (Array(2, 3, 4), Array(3, 4, 5)),
      (Array(3, 4, 5), Array(4, 5, 6)),
      (Array(4, 5, 6), Array(5, 6, 7)),
      (Array(5, 6, 7), Array(6, 7, 8)),
      (Array(6, 7, 8), Array(7, 8, 9))
    )

    val result = SlidingWindowUtil.generateInputTargetPairs(tokenIds, windowSize, stride)

    // Compare the result using sameElements
    assert(result.length == expectedPairs.length, "Length of the result does not match expected length")

    result.zip(expectedPairs).foreach { case ((input, target), (expectedInput, expectedTarget)) =>
      assert(input.sameElements(expectedInput), s"Input ${input.mkString(",")} does not match expected ${expectedInput.mkString(",")}")
      assert(target.sameElements(expectedTarget), s"Target ${target.mkString(",")} does not match expected ${expectedTarget.mkString(",")}")
    }
  }


  test("generateInputTargetPairs should return empty sequence when no valid pairs are found") {
    val tokenIds = Array(1, 2)
    val windowSize = 3
    val stride = 1

    val result = SlidingWindowUtil.generateInputTargetPairs(tokenIds, windowSize, stride)
    assert(result.isEmpty, "Expected an empty sequence for invalid window size")
  }

}


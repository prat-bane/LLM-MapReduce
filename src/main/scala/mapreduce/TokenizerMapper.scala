package mapreduce

import org.apache.hadoop.io._
import org.apache.hadoop.mapreduce._
import org.slf4j.LoggerFactory

class TokenizerMapper extends Mapper[LongWritable, Text, LongWritable, Text] {
  private val logger = LoggerFactory.getLogger(this.getClass)
  override def map(
                    key: LongWritable,
                    value: Text,
                    context: Mapper[LongWritable, Text, LongWritable, Text]#Context
                  ): Unit = {
    val line = value.toString.trim
    val splitIndex = line.indexOf('_')
    if (splitIndex > 0 && splitIndex < line.length - 1) {
      val positionStr = line.substring(0, splitIndex)
      val word = line.substring(splitIndex + 1)
      try {
        val position = positionStr.toLong
        logger.debug(s"Emitted key-value pair: ($position, $word)")
        context.write(new LongWritable(position), new Text(word))

      } catch {
        case _: NumberFormatException =>
          logger.error("Invalid position format in line")
      }
    }
  }
}

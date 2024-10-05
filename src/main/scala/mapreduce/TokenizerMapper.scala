package mapreduce

import org.apache.hadoop.io._
import org.apache.hadoop.mapreduce._

class TokenizerMapper extends Mapper[LongWritable, Text, LongWritable, Text] {
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
        context.write(new LongWritable(position), new Text(word))
      } catch {
        case _: NumberFormatException =>
        // Handle invalid position format if necessary
      }
    }
  }
}

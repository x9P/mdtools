import java.io.IOException;
import java.util.StringTokenizer;
import java.util.HashMap;
import java.util.Map;
import java.util.Iterator;
import java.lang.Math;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.ArrayWritable;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import edu.ucsc.srl.damasc.netcdf.io.input.NetCDFFileInputFormat;
import edu.ucsc.srl.damasc.netcdf.Utils;
import edu.ucsc.srl.damasc.netcdf.Utils.Operator;

import ucar.ma2.Array;
import ucar.ma2.ArrayFloat;
import ucar.ma2.ArrayDouble;
import ucar.ma2.IndexIterator;
import ucar.ma2.InvalidRangeException;
import edu.ucsc.srl.damasc.netcdf.io.ArraySpec;
import edu.ucsc.srl.damasc.netcdf.io.GroupID;
import edu.ucsc.srl.damasc.netcdf.io.GroupIDGen;
import edu.ucsc.srl.damasc.netcdf.Utils;



public class GetDistance {
  private class FloatArrayWritable extends ArrayWritable {
      /**
       * creates a new ArrayWritable of FloatWritable objects 
       */
      public FloatArrayWritable() {
          super(FloatWritable.class);
      }
      /**
       * creates a new ArrayWritable of FloatWritable objects 
       *
       * @param values array of FloatWritable objects to initialize ArrayWritable
       */
      public FloatArrayWritable(FloatWritable[] values) {
          super(FloatWritable.class, values);
      }
  }


  public static class CoordMapper
       extends Mapper<ArraySpec, Array, LongWritable, FloatWritable>{
       //extends Mapper<ArraySpec, Array, LongWritable, FloatArrayWritable>{

    private LongWritable index = new LongWritable();
    private FloatWritable r = new FloatWritable();

    // we have to extend GroupIDGen... add support for ArrayFloat data
    private static class GroupIDGenFloat extends GroupIDGen {
      // had to copy getGIDArraySpec only so my custom pullOutSubArrays could call it
      /**
       * This method creates an Array spec that is the intersection of 
       * the current GroupID (normalized into global space) and the actual
       * array being tiled over. The returned value will be smaller than or equal
       * to the extraction shape for any given dimension.
       * @param spec The corner / shape of the data we want
       * @param gid The GroupID for the current tiling of the extraction shape
       * @param extractionShape the extraction shape
       * @param ones an array of all ones (needed by calls in this function)
       * @return returns an ArraySpec object representing the data 
      */
      public static ArraySpec getGIDArraySpec( ArraySpec spec, 
						                       GroupID gid, int[] extractionShape, 
						                       int[] ones) {

          int[] normalizedGID = new int[extractionShape.length];
          int[] groupID = gid.getGroupID();

          // project the groupID into the global space, via extractionShape, 
          // and then add the startOffset
          for( int i=0; i < normalizedGID.length; i++) {
              normalizedGID[i] = (groupID[i] * extractionShape[i]); 
          }
          //System.out.println("normalized: " + Utils.arrayToString(normalizedGID));
          
          // we're going to adjust normaliedGID to be the corner of the subArray
          // need a new int[] for the shape of the subArray
          
          int[] newShape = new int[normalizedGID.length];

          // roll through the various dimensions, creating the correct corner / shape
          // pair for this GroupID (gid)
          for( int i=0; i < normalizedGID.length; i++) {
              // this gid starts prior to the data for this dimension
              newShape[i] = extractionShape[i];
              //newShape[i] =e 1;

              // in this dimension, if spec.getCorner is > normalizedGID,
              // then we need to shrink the shape accordingly.
              // Also, move the normalizedGID to match
              if ( normalizedGID[i] < spec.getCorner()[i]) { 
                  newShape[i] = extractionShape[i] - (spec.getCorner()[i] - normalizedGID[i]);
                  normalizedGID[i] = spec.getCorner()[i];
              // now, if the shape extends past the spec, shorten it again
              } else if ((normalizedGID[i] + extractionShape[i]) > 
                  (spec.getCorner()[i] + spec.getShape()[i]) ){
                newShape[i] = newShape[i] - 
                              ((normalizedGID[i] + extractionShape[i]) - 
                              (spec.getCorner()[i] + spec.getShape()[i]));
              } 
          }

          // now we need to make sure this doesn't exceed the shape of the array
          // we're working off of
          for( int i=0; i < normalizedGID.length; i++) {
            if( newShape[i] > spec.getShape()[i] )  {
              newShape[i] = spec.getShape()[i];
            }
          }


          //System.out.println("\tcorner: " + Utils.arrayToString(normalizedGID) + 
          //                   " shape: " + Utils.arrayToString(newShape) );
          
          ArraySpec returnSpec = null; 
          try {  
              returnSpec = new ArraySpec(normalizedGID, newShape, "_", "_"); 
          } catch ( Exception e ) {
              System.out.println("Caught an exception in GroupIDGen.getGIDArraySpec()");
          }

          return returnSpec;
      }

      /**
       * This should take an array, get the bounding GroupIDs.
       * <p>
       * This version works for ArrayFloat data, rather than ArrayInt in SciHadoop
       * @param myGIDG a GroupIDGen object
       * @param data the data covered by the group of produced IDs
       * @param spec the logical space to get GroupIDs for
       * @param extractionShape the shape to be tiled over the logical data space
       * @param ones helper array that is full of ones
       * @param returnMap a HashMap of GroupID to Array mappings. This carries the 
       * results of this function.
       */
      public static void pullOutSubArrays( GroupIDGen myGIDG, ArrayFloat data, // changed from ArrayInt
                                           ArraySpec spec, int[] extractionShape,
                                           int[] ones,
                                           HashMap<GroupID, Array> returnMap) {

          /*LOG.info("pullOutSubArrays passed ArraySpec: " + spec.toString() +
               " ex shape: " + Utils.arrayToString(extractionShape));*/
          GroupID[] gids = myGIDG.getBoundingGroupIDs( spec, extractionShape);
          /*LOG.info("getBoundingGroupIDs: getBounding returned " + gids.length + " gids " +
               " for ArraySpec: " + spec.toString() +
               " ex shape: " + Utils.arrayToString(extractionShape));*/

          ArraySpec tempArraySpec;
          ArrayFloat tempFloatArray;    // changed form ArrayInt
          int[] readCorner = new int[extractionShape.length];

          for( GroupID gid : gids ) {
                  //System.out.println("gid: " + gid);

                  tempArraySpec = getGIDArraySpec( spec, gid, extractionShape, ones);
                  try {

                  // note, the tempArraySpec is in the global space
                  // need to translate that into the local space prior to pull out the subarray
                  for( int i=0; i<readCorner.length; i++) {
                      readCorner[i] = tempArraySpec.getCorner()[i] - spec.getCorner()[i];
                  }

                  //System.out.println("\t\tlocal read corner: " + Utils.arrayToString(readCorner) );
                  tempFloatArray = (ArrayFloat)data.sectionNoReduce(readCorner, tempArraySpec.getShape(), ones);

                 /* 
                  System.out.println("\tsubArray ( gid: " + gid.toString(extractionShape) + 
                         " ex shape: " + Utils.arrayToString(extractionShape) + ")" + 
                         " read corner: " + Utils.arrayToString(readCorner) + 
                         " read shape: " + Utils.arrayToString(tempArraySpec.getShape()) + 
                         "\n"); 
                 */

                  returnMap.put(gid, tempFloatArray);

              } catch (InvalidRangeException ire) {
                  System.out.println("Caught an ire in GroupIDGen.pullOutSubArrays()");
              }
          }

          return;
      }
    }

    public void map(ArraySpec key, Array value, Context context
                    ) throws IOException, InterruptedException {
      ArrayFloat ncArray = (ArrayFloat)value;
      Configuration conf = context.getConfiguration();
      
      int[] match = new int[2];
      match[0] = (Integer.parseInt(conf.get("match1"))-1)*3;
      match[1] = (Integer.parseInt(conf.get("match2"))-1)*3;

      int[] variableShape =
           Utils.getVariableShape(conf);

      // need to extract xyz coordinates (1 frame, variableShape[1] atoms, 3 spatial)
      conf.set("damasc.extraction_shape", String.format("1,%d,3", variableShape[1]));
      int[] extractionShape =
          Utils.getExtractionShape(conf,
                                   key.getShape().length);
      
      int[] allOnes = new int[key.getShape().length];
      for( int i=0; i<allOnes.length; i++) {
        allOnes[i] = 1;
      }

      // boilerplate for getting out a set of coordinates from NetCDF files
      HashMap<GroupID, Array> groupSubArrayMap = new HashMap<GroupID, Array>();
      GroupIDGen myGIDG = new GroupIDGen();
      GroupIDGenFloat.pullOutSubArrays( myGIDG, ncArray, key, extractionShape,
                                        allOnes, groupSubArrayMap);

      
      Iterator<Map.Entry<GroupID, Array>> gidItr =
            groupSubArrayMap.entrySet().iterator();
            
      LongWritable myLongW = new LongWritable();
      FloatWritable myFloatW = new FloatWritable();

      GroupID localGID = new GroupID();
      localGID.setName(key.getVarName());

      ArrayFloat localArray;


      while (gidItr.hasNext() ) {
        // get next pair, split up
        Map.Entry<GroupID, Array> pairs = gidItr.next();
        localGID = pairs.getKey();
        localArray = (ArrayFloat)pairs.getValue();

        // make sure key is appropriate for global context
        Utils.adjustGIDForLogicalOffset(localGID, key.getLogicalStartOffset(), extractionShape );

        // get frame number and distance, write to context
        myLongW.set( localGID.getGroupID()[0] );
        double dx = localArray.getFloat(match[0]  ) - localArray.getFloat(match[1]  );
        double dy = localArray.getFloat(match[0]+1) - localArray.getFloat(match[1]+1);
        double dz = localArray.getFloat(match[0]+2) - localArray.getFloat(match[1]+2);
        myFloatW.set( (float)Math.sqrt(dx*dx + dy*dy + dz*dz) );

        context.write(myLongW, myFloatW);

      }

    }
  }

  public static class CoordReducer
       extends Reducer<LongWritable,FloatWritable,LongWritable,FloatWritable> {

    public void reduce(LongWritable key, Iterable<FloatWritable> values,
                       Context context
                       ) throws IOException, InterruptedException {
      // this could be more sophisticated (mean, variance, whatever)
      // here, we just print the distances back out...
      for (FloatWritable val : values)
        context.write(key, val);
    }
  }

  public static void main(String[] args) throws Exception {
    Configuration conf = new Configuration();

    // set some configuration variables
    conf.set("damasc.variable_name", "coordinates");

    // want to get distances between atoms in arguments 2 and 3
    conf.set("match1", args[2]);
    conf.set("match2", args[3]);

    // tell MR what to do
    Job job = Job.getInstance(conf, "GetDistance");
    job.setJarByClass(GetDistance.class);
    job.setMapperClass(CoordMapper.class);
    job.setCombinerClass(CoordReducer.class);
    job.setReducerClass(CoordReducer.class);
    job.setOutputKeyClass(LongWritable.class);
    job.setOutputValueClass(FloatWritable.class);
    job.setInputFormatClass(NetCDFFileInputFormat.class);
    
    // arguments 0 and 1 specify input and output paths
    NetCDFFileInputFormat.addInputPath(job, new Path(args[0]));
    FileOutputFormat.setOutputPath(job, new Path(args[1]));
    
    System.exit(job.waitForCompletion(true) ? 0 : 1);
  }
}

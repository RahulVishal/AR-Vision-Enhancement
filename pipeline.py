import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GstApp
import numpy as np
import cv2
import gc

# Initialize GStreamer
Gst.init(None)

# Custom image processing function (modify this)
def process_frame(frame):
    # Example: Convert to grayscale
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# GStreamer pipeline with appsink
pipeline_str = (
    'udpsrc port=5000 caps="application/x-rtp, media=video, encoding-name=H264, payload=96" ! '
    'rtph264depay ! avdec_h264 ! videoconvert ! video/x-raw,format=BGR ! appsink name=sink emit-signals=true max-buffers=1 drop=true'
)

# Create pipeline
pipeline = Gst.parse_launch(pipeline_str)
appsink = pipeline.get_by_name("sink")

# Configure appsink
appsink.set_property("emit-signals", True)
appsink.set_property("sync", False)

# Callback for new sample
def on_new_sample(sink):
    sample = sink.emit("pull-sample")
    if sample:
        buf = sample.get_buffer()
        caps = sample.get_caps()
        width = caps.get_structure(0).get_value('width')
        height = caps.get_structure(0).get_value('height')

        # Extract frame data from buffer
        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.frombuffer(map_info.data, np.uint8).reshape((height, width, 3))
        buf.unmap(map_info)

        # Apply custom image processing
        processed = process_frame(frame)

        # Show the result (optional)
        cv2.imshow("Processed Frame", processed)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            loop.quit()

        # Clean up memory
        del frame
        del processed
        gc.collect()

    return Gst.FlowReturn.OK

# Connect signal
appsink.connect("new-sample", on_new_sample)

# Run pipeline
pipeline.set_state(Gst.State.PLAYING)
loop = GLib.MainLoop()

# Setup bus for error handling
bus = pipeline.get_bus()
bus.add_signal_watch()

def on_message(bus, message):
    t = message.type
    if t == Gst.MessageType.EOS:
        print("End-of-stream")
        loop.quit()
    elif t == Gst.MessageType.ERROR:
        err, debug = message.parse_error()
        print(f"Error: {err.message}")
        if debug:
            print(f"Debug: {debug}")
        loop.quit()

bus.connect("message", on_message)

try:
    print("Streaming and processing frames... Press 'q' to quit.")
    loop.run()
except KeyboardInterrupt:
    print("Interrupted by user.")
finally:
    pipeline.set_state(Gst.State.NULL)
    cv2.destroyAllWindows()
Several of the scripts in this repo are instrumented to aid in debugging using Instruments. If you haven't already, install Xcode which comes with the Instruments app.

## Enable the CoreML Instrument
The Instruments app comes with a CoreML instrument that shows details about how long it takes to load and perform predictions on CoreML models. To make this work with Python you'll need to re-sign your Python binary.

Make a file `entitlements.xml` with the following contents:
```xml
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
        <key>com.apple.security.get-task-allow</key>
        <true/>
</dict>
</plist>
```

Find your Python binary. It very likely is not what is returned by `which python`. If you installed it via homebrew it's likely in a path like this:

/opt/homebrew/Cellar/python@3.10/3.10.12_1/Frameworks/Python.framework/Versions/3.10/Resources/Python.app/Contents/MacOS/Python

or if you installed via Xcode:

/Applications/Xcode-14.3.1.app/Contents/Developer/Library/Frameworks/Python3.framework/Versions/3.9/Resources/Python.app/Contents/MacOS/Python

Re-sign that binary to grant it the get-task-allow entitlement

```shell
codesign --force --sign - /opt/homebrew/Cellar/python@3.10/3.10.12_1/Frameworks/Python.framework/Versions/3.10/Resources/Python.app/Contents/MacOS/Python --entitlements entitlements.xml
```

## Record and View a Trace
You can record a trace using the Instruments UI, but it's easier to run via the command line:

```shell
xctrace record --template InstrumentsTemplate.tracetemplate --launch -- /path/to/python generate.py --compute_unit CPUAndANE --model_path gpt2.mlpackage
```

This will run until the program exits or until you hit Ctrl-C. Open the file in Instruments. This should give you a good overview of what happened wile the script ran. Be sure to expand the os_signpost instrument to see the spans for how long each prediction took.
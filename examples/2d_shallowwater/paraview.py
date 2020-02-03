# trace generated using paraview version 5.6.2
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
x_u_test_pred_mean_stdvtu = XMLUnstructuredGridReader(FileName=['/Users/pierremtb/LocalWork/ETS/POD-NN/examples/2d_shallowwater/cache/x_u_test_pred_mean_std.vtu'])
x_u_test_pred_mean_stdvtu.PointArrayStatus = ['h_mean', 'hu_mean', 'hv_mean', 'h_std', 'hu_std', 'hv_std', 'h_mean_pred', 'hu_mean_pred', 'hv_mean_pred', 'h_std_pred', 'hu_std_pred', 'hv_std_pred']

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [2197, 818]

# show data in view
x_u_test_pred_mean_stdvtuDisplay = Show(x_u_test_pred_mean_stdvtu, renderView1)

# trace defaults for the display properties.
x_u_test_pred_mean_stdvtuDisplay.Representation = 'Surface'
x_u_test_pred_mean_stdvtuDisplay.ColorArrayName = [None, '']
x_u_test_pred_mean_stdvtuDisplay.OSPRayScaleArray = 'h_mean'
x_u_test_pred_mean_stdvtuDisplay.OSPRayScaleFunction = 'PiecewiseFunction'
x_u_test_pred_mean_stdvtuDisplay.SelectOrientationVectors = 'None'
x_u_test_pred_mean_stdvtuDisplay.ScaleFactor = 906.1500000000001
x_u_test_pred_mean_stdvtuDisplay.SelectScaleArray = 'None'
x_u_test_pred_mean_stdvtuDisplay.GlyphType = 'Arrow'
x_u_test_pred_mean_stdvtuDisplay.GlyphTableIndexArray = 'None'
x_u_test_pred_mean_stdvtuDisplay.GaussianRadius = 45.3075
x_u_test_pred_mean_stdvtuDisplay.SetScaleArray = ['POINTS', 'h_mean']
x_u_test_pred_mean_stdvtuDisplay.ScaleTransferFunction = 'PiecewiseFunction'
x_u_test_pred_mean_stdvtuDisplay.OpacityArray = ['POINTS', 'h_mean']
x_u_test_pred_mean_stdvtuDisplay.OpacityTransferFunction = 'PiecewiseFunction'
x_u_test_pred_mean_stdvtuDisplay.DataAxesGrid = 'GridAxesRepresentation'
x_u_test_pred_mean_stdvtuDisplay.SelectionCellLabelFontFile = ''
x_u_test_pred_mean_stdvtuDisplay.SelectionPointLabelFontFile = ''
x_u_test_pred_mean_stdvtuDisplay.PolarAxes = 'PolarAxesRepresentation'
x_u_test_pred_mean_stdvtuDisplay.ScalarOpacityUnitDistance = 158.7299088688611

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
x_u_test_pred_mean_stdvtuDisplay.DataAxesGrid.XTitleFontFile = ''
x_u_test_pred_mean_stdvtuDisplay.DataAxesGrid.YTitleFontFile = ''
x_u_test_pred_mean_stdvtuDisplay.DataAxesGrid.ZTitleFontFile = ''
x_u_test_pred_mean_stdvtuDisplay.DataAxesGrid.XLabelFontFile = ''
x_u_test_pred_mean_stdvtuDisplay.DataAxesGrid.YLabelFontFile = ''
x_u_test_pred_mean_stdvtuDisplay.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
x_u_test_pred_mean_stdvtuDisplay.PolarAxes.PolarAxisTitleFontFile = ''
x_u_test_pred_mean_stdvtuDisplay.PolarAxes.PolarAxisLabelFontFile = ''
x_u_test_pred_mean_stdvtuDisplay.PolarAxes.LastRadialAxisTextFontFile = ''
x_u_test_pred_mean_stdvtuDisplay.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# reset view to fit data
renderView1.ResetCamera()

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(Input=x_u_test_pred_mean_stdvtu,
    Source='High Resolution Line Source')

# init the 'High Resolution Line Source' selected for 'Source'
plotOverLine1.Source.Point1 = [271699.25, 5041580.5, 0.0]
plotOverLine1.Source.Point2 = [280229.2812, 5050642.0, 0.0]

# show data in view
plotOverLine1Display = Show(plotOverLine1, renderView1)

# trace defaults for the display properties.
plotOverLine1Display.Representation = 'Surface'
plotOverLine1Display.ColorArrayName = [None, '']
plotOverLine1Display.OSPRayScaleArray = 'arc_length'
plotOverLine1Display.OSPRayScaleFunction = 'PiecewiseFunction'
plotOverLine1Display.SelectOrientationVectors = 'None'
plotOverLine1Display.ScaleFactor = 906.1500000000001
plotOverLine1Display.SelectScaleArray = 'None'
plotOverLine1Display.GlyphType = 'Arrow'
plotOverLine1Display.GlyphTableIndexArray = 'None'
plotOverLine1Display.GaussianRadius = 45.3075
plotOverLine1Display.SetScaleArray = ['POINTS', 'arc_length']
plotOverLine1Display.ScaleTransferFunction = 'PiecewiseFunction'
plotOverLine1Display.OpacityArray = ['POINTS', 'arc_length']
plotOverLine1Display.OpacityTransferFunction = 'PiecewiseFunction'
plotOverLine1Display.DataAxesGrid = 'GridAxesRepresentation'
plotOverLine1Display.SelectionCellLabelFontFile = ''
plotOverLine1Display.SelectionPointLabelFontFile = ''
plotOverLine1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
plotOverLine1Display.DataAxesGrid.XTitleFontFile = ''
plotOverLine1Display.DataAxesGrid.YTitleFontFile = ''
plotOverLine1Display.DataAxesGrid.ZTitleFontFile = ''
plotOverLine1Display.DataAxesGrid.XLabelFontFile = ''
plotOverLine1Display.DataAxesGrid.YLabelFontFile = ''
plotOverLine1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
plotOverLine1Display.PolarAxes.PolarAxisTitleFontFile = ''
plotOverLine1Display.PolarAxes.PolarAxisLabelFontFile = ''
plotOverLine1Display.PolarAxes.LastRadialAxisTextFontFile = ''
plotOverLine1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')
lineChartView1.ViewSize = [1094, 818]
lineChartView1.ChartTitleFontFile = ''
lineChartView1.LeftAxisTitleFontFile = ''
lineChartView1.LeftAxisRangeMaximum = 6.66
lineChartView1.LeftAxisLabelFontFile = ''
lineChartView1.BottomAxisTitleFontFile = ''
lineChartView1.BottomAxisRangeMaximum = 6.66
lineChartView1.BottomAxisLabelFontFile = ''
lineChartView1.RightAxisRangeMaximum = 6.66
lineChartView1.RightAxisLabelFontFile = ''
lineChartView1.TopAxisTitleFontFile = ''
lineChartView1.TopAxisRangeMaximum = 6.66
lineChartView1.TopAxisLabelFontFile = ''

# get layout
layout1 = GetLayout()

# place view in the layout
layout1.AssignView(2, lineChartView1)

# show data in view
plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1)

# trace defaults for the display properties.
plotOverLine1Display_1.CompositeDataSetIndex = [0]
plotOverLine1Display_1.UseIndexForXAxis = 0
plotOverLine1Display_1.XArrayName = 'arc_length'
plotOverLine1Display_1.SeriesVisibility = ['h_mean', 'h_mean_pred', 'h_std', 'h_std_pred', 'hu_mean', 'hu_mean_pred', 'hu_std', 'hu_std_pred', 'hv_mean', 'hv_mean_pred', 'hv_std', 'hv_std_pred']
plotOverLine1Display_1.SeriesLabel = ['arc_length', 'arc_length', 'h_mean', 'h_mean', 'h_mean_pred', 'h_mean_pred', 'h_std', 'h_std', 'h_std_pred', 'h_std_pred', 'hu_mean', 'hu_mean', 'hu_mean_pred', 'hu_mean_pred', 'hu_std', 'hu_std', 'hu_std_pred', 'hu_std_pred', 'hv_mean', 'hv_mean', 'hv_mean_pred', 'hv_mean_pred', 'hv_std', 'hv_std', 'hv_std_pred', 'hv_std_pred', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine1Display_1.SeriesColor = ['arc_length', '0', '0', '0', 'h_mean', '0.89', '0.1', '0.11', 'h_mean_pred', '0.22', '0.49', '0.72', 'h_std', '0.3', '0.69', '0.29', 'h_std_pred', '0.6', '0.31', '0.64', 'hu_mean', '1', '0.5', '0', 'hu_mean_pred', '0.65', '0.34', '0.16', 'hu_std', '0', '0', '0', 'hu_std_pred', '0.89', '0.1', '0.11', 'hv_mean', '0.22', '0.49', '0.72', 'hv_mean_pred', '0.3', '0.69', '0.29', 'hv_std', '0.6', '0.31', '0.64', 'hv_std_pred', '1', '0.5', '0', 'vtkValidPointMask', '0.65', '0.34', '0.16', 'Points_X', '0', '0', '0', 'Points_Y', '0.89', '0.1', '0.11', 'Points_Z', '0.22', '0.49', '0.72', 'Points_Magnitude', '0.3', '0.69', '0.29']
plotOverLine1Display_1.SeriesPlotCorner = ['arc_length', '0', 'h_mean', '0', 'h_mean_pred', '0', 'h_std', '0', 'h_std_pred', '0', 'hu_mean', '0', 'hu_mean_pred', '0', 'hu_std', '0', 'hu_std_pred', '0', 'hv_mean', '0', 'hv_mean_pred', '0', 'hv_std', '0', 'hv_std_pred', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine1Display_1.SeriesLabelPrefix = ''
plotOverLine1Display_1.SeriesLineStyle = ['arc_length', '1', 'h_mean', '1', 'h_mean_pred', '1', 'h_std', '1', 'h_std_pred', '1', 'hu_mean', '1', 'hu_mean_pred', '1', 'hu_std', '1', 'hu_std_pred', '1', 'hv_mean', '1', 'hv_mean_pred', '1', 'hv_std', '1', 'hv_std_pred', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine1Display_1.SeriesLineThickness = ['arc_length', '2', 'h_mean', '2', 'h_mean_pred', '2', 'h_std', '2', 'h_std_pred', '2', 'hu_mean', '2', 'hu_mean_pred', '2', 'hu_std', '2', 'hu_std_pred', '2', 'hv_mean', '2', 'hv_mean_pred', '2', 'hv_std', '2', 'hv_std_pred', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
plotOverLine1Display_1.SeriesMarkerStyle = ['arc_length', '0', 'h_mean', '0', 'h_mean_pred', '0', 'h_std', '0', 'h_std_pred', '0', 'hu_mean', '0', 'hu_mean_pred', '0', 'hu_std', '0', 'hu_std_pred', '0', 'hv_mean', '0', 'hv_mean_pred', '0', 'hv_std', '0', 'hv_std_pred', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
lineChartView1.Update()

# set active view
SetActiveView(renderView1)

# Properties modified on plotOverLine1.Source
plotOverLine1.Source.Point1 = [275091.894182153, 5043254.001614607, 0.0]
plotOverLine1.Source.Point2 = [273900.40236693545, 5044343.5484692, 0.0]

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
lineChartView1.Update()

# set active view
SetActiveView(lineChartView1)

# save data
SaveData('/Users/pierremtb/LocalWork/ETS/POD-NN/examples/2d_shallowwater/cache/x_u_test_pred_mean_std.csv', proxy=plotOverLine1)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [275964.26560000004, 5046111.25, 24041.442828591742]
renderView1.CameraFocalPoint = [275964.26560000004, 5046111.25, 0.0]
renderView1.CameraParallelScale = 6222.38327578296

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
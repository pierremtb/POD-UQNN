# trace generated using paraview version 5.7.0
#
# To ensure correct image size when batch processing, please search 
# for and uncomment the line `# renderView*.ViewSize = [*,*]`

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'XML Unstructured Grid Reader'
x_u_tst_pred_0 = XMLUnstructuredGridReader(FileName=['/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred_0.0.vtu', '/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred_0.1.vtu', '/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred_0.2.vtu', '/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred_0.3.vtu', '/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred_0.4.vtu', '/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred_0.5.vtu', '/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred_0.6.vtu', '/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred_0.7.vtu', '/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred_0.8.vtu', '/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred_0.9.vtu'])
x_u_tst_pred_0.CellArrayStatus = []
x_u_tst_pred_0.PointArrayStatus = ['eta', 'eta_pred', 'eta_pred_up', 'eta_pred_lo']

# get animation scene
animationScene1 = GetAnimationScene()

# get the time-keeper
timeKeeper1 = GetTimeKeeper()

layout1 = GetLayout()

# update animation scene based on data timesteps
animationScene1.UpdateAnimationUsingDataTimeSteps()

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [917, 818]

# show data in view
x_u_tst_pred_0Display = Show(x_u_tst_pred_0, renderView1)

# trace defaults for the display properties.
x_u_tst_pred_0Display.Representation = 'Surface'
x_u_tst_pred_0Display.AmbientColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.ColorArrayName = [None, '']
x_u_tst_pred_0Display.DiffuseColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.LookupTable = None
x_u_tst_pred_0Display.MapScalars = 1
x_u_tst_pred_0Display.MultiComponentsMapping = 0
x_u_tst_pred_0Display.InterpolateScalarsBeforeMapping = 1
x_u_tst_pred_0Display.Opacity = 1.0
x_u_tst_pred_0Display.PointSize = 2.0
x_u_tst_pred_0Display.LineWidth = 1.0
x_u_tst_pred_0Display.RenderLinesAsTubes = 0
x_u_tst_pred_0Display.RenderPointsAsSpheres = 0
x_u_tst_pred_0Display.Interpolation = 'Gouraud'
x_u_tst_pred_0Display.Specular = 0.0
x_u_tst_pred_0Display.SpecularColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.SpecularPower = 100.0
x_u_tst_pred_0Display.Luminosity = 0.0
x_u_tst_pred_0Display.Ambient = 0.0
x_u_tst_pred_0Display.Diffuse = 1.0
x_u_tst_pred_0Display.EdgeColor = [0.0, 0.0, 0.5]
x_u_tst_pred_0Display.BackfaceRepresentation = 'Follow Frontface'
x_u_tst_pred_0Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.BackfaceOpacity = 1.0
x_u_tst_pred_0Display.Position = [0.0, 0.0, 0.0]
x_u_tst_pred_0Display.Scale = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.Orientation = [0.0, 0.0, 0.0]
x_u_tst_pred_0Display.Origin = [0.0, 0.0, 0.0]
x_u_tst_pred_0Display.Pickable = 1
x_u_tst_pred_0Display.Texture = None
x_u_tst_pred_0Display.Triangulate = 0
x_u_tst_pred_0Display.UseShaderReplacements = 0
x_u_tst_pred_0Display.ShaderReplacements = ''
x_u_tst_pred_0Display.NonlinearSubdivisionLevel = 1
x_u_tst_pred_0Display.UseDataPartitions = 0
x_u_tst_pred_0Display.OSPRayUseScaleArray = 0
x_u_tst_pred_0Display.OSPRayScaleArray = 'eta'
x_u_tst_pred_0Display.OSPRayScaleFunction = 'PiecewiseFunction'
x_u_tst_pred_0Display.OSPRayMaterial = 'None'
x_u_tst_pred_0Display.Orient = 0
x_u_tst_pred_0Display.OrientationMode = 'Direction'
x_u_tst_pred_0Display.SelectOrientationVectors = 'None'
x_u_tst_pred_0Display.Scaling = 0
x_u_tst_pred_0Display.ScaleMode = 'No Data Scaling Off'
x_u_tst_pred_0Display.ScaleFactor = 56.550000000000004
x_u_tst_pred_0Display.SelectScaleArray = 'None'
x_u_tst_pred_0Display.GlyphType = 'Arrow'
x_u_tst_pred_0Display.UseGlyphTable = 0
x_u_tst_pred_0Display.GlyphTableIndexArray = 'None'
x_u_tst_pred_0Display.UseCompositeGlyphTable = 0
x_u_tst_pred_0Display.UseGlyphCullingAndLOD = 0
x_u_tst_pred_0Display.LODValues = []
x_u_tst_pred_0Display.ColorByLODIndex = 0
x_u_tst_pred_0Display.GaussianRadius = 2.8275
x_u_tst_pred_0Display.ShaderPreset = 'Sphere'
x_u_tst_pred_0Display.CustomTriangleScale = 3
x_u_tst_pred_0Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
x_u_tst_pred_0Display.Emissive = 0
x_u_tst_pred_0Display.ScaleByArray = 0
x_u_tst_pred_0Display.SetScaleArray = ['POINTS', 'eta']
x_u_tst_pred_0Display.ScaleArrayComponent = ''
x_u_tst_pred_0Display.UseScaleFunction = 1
x_u_tst_pred_0Display.ScaleTransferFunction = 'PiecewiseFunction'
x_u_tst_pred_0Display.OpacityByArray = 0
x_u_tst_pred_0Display.OpacityArray = ['POINTS', 'eta']
x_u_tst_pred_0Display.OpacityArrayComponent = ''
x_u_tst_pred_0Display.OpacityTransferFunction = 'PiecewiseFunction'
x_u_tst_pred_0Display.DataAxesGrid = 'GridAxesRepresentation'
x_u_tst_pred_0Display.SelectionCellLabelBold = 0
x_u_tst_pred_0Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
x_u_tst_pred_0Display.SelectionCellLabelFontFamily = 'Arial'
x_u_tst_pred_0Display.SelectionCellLabelFontFile = ''
x_u_tst_pred_0Display.SelectionCellLabelFontSize = 18
x_u_tst_pred_0Display.SelectionCellLabelItalic = 0
x_u_tst_pred_0Display.SelectionCellLabelJustification = 'Left'
x_u_tst_pred_0Display.SelectionCellLabelOpacity = 1.0
x_u_tst_pred_0Display.SelectionCellLabelShadow = 0
x_u_tst_pred_0Display.SelectionPointLabelBold = 0
x_u_tst_pred_0Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
x_u_tst_pred_0Display.SelectionPointLabelFontFamily = 'Arial'
x_u_tst_pred_0Display.SelectionPointLabelFontFile = ''
x_u_tst_pred_0Display.SelectionPointLabelFontSize = 18
x_u_tst_pred_0Display.SelectionPointLabelItalic = 0
x_u_tst_pred_0Display.SelectionPointLabelJustification = 'Left'
x_u_tst_pred_0Display.SelectionPointLabelOpacity = 1.0
x_u_tst_pred_0Display.SelectionPointLabelShadow = 0
x_u_tst_pred_0Display.PolarAxes = 'PolarAxesRepresentation'
x_u_tst_pred_0Display.ScalarOpacityFunction = None
x_u_tst_pred_0Display.ScalarOpacityUnitDistance = 30.24957638082656
x_u_tst_pred_0Display.ExtractedBlockIndex = 0
x_u_tst_pred_0Display.SelectMapper = 'Projected tetra'
x_u_tst_pred_0Display.SamplingDimensions = [128, 128, 128]
x_u_tst_pred_0Display.UseFloatingPointFrameBuffer = 1

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
x_u_tst_pred_0Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
x_u_tst_pred_0Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
x_u_tst_pred_0Display.GlyphType.TipResolution = 6
x_u_tst_pred_0Display.GlyphType.TipRadius = 0.1
x_u_tst_pred_0Display.GlyphType.TipLength = 0.35
x_u_tst_pred_0Display.GlyphType.ShaftResolution = 6
x_u_tst_pred_0Display.GlyphType.ShaftRadius = 0.03
x_u_tst_pred_0Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
x_u_tst_pred_0Display.ScaleTransferFunction.Points = [28.799999237060547, 0.0, 0.5, 0.0, 35.063018798828125, 1.0, 0.5, 0.0]
x_u_tst_pred_0Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
x_u_tst_pred_0Display.OpacityTransferFunction.Points = [28.799999237060547, 0.0, 0.5, 0.0, 35.063018798828125, 1.0, 0.5, 0.0]
x_u_tst_pred_0Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
x_u_tst_pred_0Display.DataAxesGrid.XTitle = 'X Axis'
x_u_tst_pred_0Display.DataAxesGrid.YTitle = 'Y Axis'
x_u_tst_pred_0Display.DataAxesGrid.ZTitle = 'Z Axis'
x_u_tst_pred_0Display.DataAxesGrid.XTitleColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.DataAxesGrid.XTitleFontFamily = 'Arial'
x_u_tst_pred_0Display.DataAxesGrid.XTitleFontFile = ''
x_u_tst_pred_0Display.DataAxesGrid.XTitleBold = 0
x_u_tst_pred_0Display.DataAxesGrid.XTitleItalic = 0
x_u_tst_pred_0Display.DataAxesGrid.XTitleFontSize = 12
x_u_tst_pred_0Display.DataAxesGrid.XTitleShadow = 0
x_u_tst_pred_0Display.DataAxesGrid.XTitleOpacity = 1.0
x_u_tst_pred_0Display.DataAxesGrid.YTitleColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.DataAxesGrid.YTitleFontFamily = 'Arial'
x_u_tst_pred_0Display.DataAxesGrid.YTitleFontFile = ''
x_u_tst_pred_0Display.DataAxesGrid.YTitleBold = 0
x_u_tst_pred_0Display.DataAxesGrid.YTitleItalic = 0
x_u_tst_pred_0Display.DataAxesGrid.YTitleFontSize = 12
x_u_tst_pred_0Display.DataAxesGrid.YTitleShadow = 0
x_u_tst_pred_0Display.DataAxesGrid.YTitleOpacity = 1.0
x_u_tst_pred_0Display.DataAxesGrid.ZTitleColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
x_u_tst_pred_0Display.DataAxesGrid.ZTitleFontFile = ''
x_u_tst_pred_0Display.DataAxesGrid.ZTitleBold = 0
x_u_tst_pred_0Display.DataAxesGrid.ZTitleItalic = 0
x_u_tst_pred_0Display.DataAxesGrid.ZTitleFontSize = 12
x_u_tst_pred_0Display.DataAxesGrid.ZTitleShadow = 0
x_u_tst_pred_0Display.DataAxesGrid.ZTitleOpacity = 1.0
x_u_tst_pred_0Display.DataAxesGrid.FacesToRender = 63
x_u_tst_pred_0Display.DataAxesGrid.CullBackface = 0
x_u_tst_pred_0Display.DataAxesGrid.CullFrontface = 1
x_u_tst_pred_0Display.DataAxesGrid.GridColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.DataAxesGrid.ShowGrid = 0
x_u_tst_pred_0Display.DataAxesGrid.ShowEdges = 1
x_u_tst_pred_0Display.DataAxesGrid.ShowTicks = 1
x_u_tst_pred_0Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
x_u_tst_pred_0Display.DataAxesGrid.AxesToLabel = 63
x_u_tst_pred_0Display.DataAxesGrid.XLabelColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.DataAxesGrid.XLabelFontFamily = 'Arial'
x_u_tst_pred_0Display.DataAxesGrid.XLabelFontFile = ''
x_u_tst_pred_0Display.DataAxesGrid.XLabelBold = 0
x_u_tst_pred_0Display.DataAxesGrid.XLabelItalic = 0
x_u_tst_pred_0Display.DataAxesGrid.XLabelFontSize = 12
x_u_tst_pred_0Display.DataAxesGrid.XLabelShadow = 0
x_u_tst_pred_0Display.DataAxesGrid.XLabelOpacity = 1.0
x_u_tst_pred_0Display.DataAxesGrid.YLabelColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.DataAxesGrid.YLabelFontFamily = 'Arial'
x_u_tst_pred_0Display.DataAxesGrid.YLabelFontFile = ''
x_u_tst_pred_0Display.DataAxesGrid.YLabelBold = 0
x_u_tst_pred_0Display.DataAxesGrid.YLabelItalic = 0
x_u_tst_pred_0Display.DataAxesGrid.YLabelFontSize = 12
x_u_tst_pred_0Display.DataAxesGrid.YLabelShadow = 0
x_u_tst_pred_0Display.DataAxesGrid.YLabelOpacity = 1.0
x_u_tst_pred_0Display.DataAxesGrid.ZLabelColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
x_u_tst_pred_0Display.DataAxesGrid.ZLabelFontFile = ''
x_u_tst_pred_0Display.DataAxesGrid.ZLabelBold = 0
x_u_tst_pred_0Display.DataAxesGrid.ZLabelItalic = 0
x_u_tst_pred_0Display.DataAxesGrid.ZLabelFontSize = 12
x_u_tst_pred_0Display.DataAxesGrid.ZLabelShadow = 0
x_u_tst_pred_0Display.DataAxesGrid.ZLabelOpacity = 1.0
x_u_tst_pred_0Display.DataAxesGrid.XAxisNotation = 'Mixed'
x_u_tst_pred_0Display.DataAxesGrid.XAxisPrecision = 2
x_u_tst_pred_0Display.DataAxesGrid.XAxisUseCustomLabels = 0
x_u_tst_pred_0Display.DataAxesGrid.XAxisLabels = []
x_u_tst_pred_0Display.DataAxesGrid.YAxisNotation = 'Mixed'
x_u_tst_pred_0Display.DataAxesGrid.YAxisPrecision = 2
x_u_tst_pred_0Display.DataAxesGrid.YAxisUseCustomLabels = 0
x_u_tst_pred_0Display.DataAxesGrid.YAxisLabels = []
x_u_tst_pred_0Display.DataAxesGrid.ZAxisNotation = 'Mixed'
x_u_tst_pred_0Display.DataAxesGrid.ZAxisPrecision = 2
x_u_tst_pred_0Display.DataAxesGrid.ZAxisUseCustomLabels = 0
x_u_tst_pred_0Display.DataAxesGrid.ZAxisLabels = []
x_u_tst_pred_0Display.DataAxesGrid.UseCustomBounds = 0
x_u_tst_pred_0Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
x_u_tst_pred_0Display.PolarAxes.Visibility = 0
x_u_tst_pred_0Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
x_u_tst_pred_0Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
x_u_tst_pred_0Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
x_u_tst_pred_0Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.EnableCustomRange = 0
x_u_tst_pred_0Display.PolarAxes.CustomRange = [0.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.PolarAxisVisibility = 1
x_u_tst_pred_0Display.PolarAxes.RadialAxesVisibility = 1
x_u_tst_pred_0Display.PolarAxes.DrawRadialGridlines = 1
x_u_tst_pred_0Display.PolarAxes.PolarArcsVisibility = 1
x_u_tst_pred_0Display.PolarAxes.DrawPolarArcsGridlines = 1
x_u_tst_pred_0Display.PolarAxes.NumberOfRadialAxes = 0
x_u_tst_pred_0Display.PolarAxes.AutoSubdividePolarAxis = 1
x_u_tst_pred_0Display.PolarAxes.NumberOfPolarAxis = 0
x_u_tst_pred_0Display.PolarAxes.MinimumRadius = 0.0
x_u_tst_pred_0Display.PolarAxes.MinimumAngle = 0.0
x_u_tst_pred_0Display.PolarAxes.MaximumAngle = 90.0
x_u_tst_pred_0Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
x_u_tst_pred_0Display.PolarAxes.Ratio = 1.0
x_u_tst_pred_0Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitleVisibility = 1
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
x_u_tst_pred_0Display.PolarAxes.PolarLabelVisibility = 1
x_u_tst_pred_0Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
x_u_tst_pred_0Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
x_u_tst_pred_0Display.PolarAxes.RadialLabelVisibility = 1
x_u_tst_pred_0Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
x_u_tst_pred_0Display.PolarAxes.RadialLabelLocation = 'Bottom'
x_u_tst_pred_0Display.PolarAxes.RadialUnitsVisibility = 1
x_u_tst_pred_0Display.PolarAxes.ScreenSize = 10.0
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitleColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitleOpacity = 1.0
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitleFontFile = ''
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitleBold = 0
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitleItalic = 0
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitleShadow = 0
x_u_tst_pred_0Display.PolarAxes.PolarAxisTitleFontSize = 12
x_u_tst_pred_0Display.PolarAxes.PolarAxisLabelColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.PolarAxisLabelOpacity = 1.0
x_u_tst_pred_0Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
x_u_tst_pred_0Display.PolarAxes.PolarAxisLabelFontFile = ''
x_u_tst_pred_0Display.PolarAxes.PolarAxisLabelBold = 0
x_u_tst_pred_0Display.PolarAxes.PolarAxisLabelItalic = 0
x_u_tst_pred_0Display.PolarAxes.PolarAxisLabelShadow = 0
x_u_tst_pred_0Display.PolarAxes.PolarAxisLabelFontSize = 12
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisTextColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisTextFontFile = ''
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisTextBold = 0
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisTextItalic = 0
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisTextShadow = 0
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisTextFontSize = 12
x_u_tst_pred_0Display.PolarAxes.SecondaryRadialAxesTextColor = [1.0, 1.0, 1.0]
x_u_tst_pred_0Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
x_u_tst_pred_0Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
x_u_tst_pred_0Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
x_u_tst_pred_0Display.PolarAxes.SecondaryRadialAxesTextBold = 0
x_u_tst_pred_0Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
x_u_tst_pred_0Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
x_u_tst_pred_0Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
x_u_tst_pred_0Display.PolarAxes.EnableDistanceLOD = 1
x_u_tst_pred_0Display.PolarAxes.DistanceLODThreshold = 0.7
x_u_tst_pred_0Display.PolarAxes.EnableViewAngleLOD = 1
x_u_tst_pred_0Display.PolarAxes.ViewAngleLODThreshold = 0.7
x_u_tst_pred_0Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
x_u_tst_pred_0Display.PolarAxes.PolarTicksVisibility = 1
x_u_tst_pred_0Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
x_u_tst_pred_0Display.PolarAxes.TickLocation = 'Both'
x_u_tst_pred_0Display.PolarAxes.AxisTickVisibility = 1
x_u_tst_pred_0Display.PolarAxes.AxisMinorTickVisibility = 0
x_u_tst_pred_0Display.PolarAxes.ArcTickVisibility = 1
x_u_tst_pred_0Display.PolarAxes.ArcMinorTickVisibility = 0
x_u_tst_pred_0Display.PolarAxes.DeltaAngleMajor = 10.0
x_u_tst_pred_0Display.PolarAxes.DeltaAngleMinor = 5.0
x_u_tst_pred_0Display.PolarAxes.PolarAxisMajorTickSize = 0.0
x_u_tst_pred_0Display.PolarAxes.PolarAxisTickRatioSize = 0.3
x_u_tst_pred_0Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
x_u_tst_pred_0Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
x_u_tst_pred_0Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
x_u_tst_pred_0Display.PolarAxes.ArcMajorTickSize = 0.0
x_u_tst_pred_0Display.PolarAxes.ArcTickRatioSize = 0.3
x_u_tst_pred_0Display.PolarAxes.ArcMajorTickThickness = 1.0
x_u_tst_pred_0Display.PolarAxes.ArcTickRatioThickness = 0.5
x_u_tst_pred_0Display.PolarAxes.Use2DMode = 0
x_u_tst_pred_0Display.PolarAxes.UseLogAxis = 0

# reset view to fit data
renderView1.ResetCamera()

#changing interaction mode based on data extents
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [274952.78125, 5043802.75, 10000.0]
renderView1.CameraFocalPoint = [274952.78125, 5043802.75, 0.0]

# get the material library
materialLibrary1 = GetMaterialLibrary()

# update the view to ensure updated data information
renderView1.Update()

# get layout

# split cell
# layout1.SplitHorizontal(0, 0.5)

# set active view
SetActiveView(None)

# set active view
SetActiveView(renderView1)


# layout1 = GetLayout()
# split cell
# layout1.SplitVertical(2, 0.5)

# set active view
SetActiveView(renderView1)

# split cell
layout1.SplitVertical(1, 0.5)

# set active view
SetActiveView(None)

# resize frame
# layout1.SetSplitFraction(0, 0.434452871073)

# set active view
SetActiveView(renderView1)

# resize frame
# layout1.SetSplitFraction(0, 0.422535211268)

# set scalar coloring
ColorBy(x_u_tst_pred_0Display, ('POINTS', 'eta'))

# rescale color and/or opacity maps used to include current data range
x_u_tst_pred_0Display.RescaleTransferFunctionToDataRange(True, False)

# show color bar/color legend
x_u_tst_pred_0Display.SetScalarBarVisibility(renderView1, True)

# get color transfer function/color map for 'eta'
etaLUT = GetColorTransferFunction('eta')
etaLUT.AutomaticRescaleRangeMode = "Grow and update on 'Apply'"
etaLUT.InterpretValuesAsCategories = 0
etaLUT.AnnotationsInitialized = 0
etaLUT.ShowCategoricalColorsinDataRangeOnly = 0
etaLUT.RescaleOnVisibilityChange = 0
etaLUT.EnableOpacityMapping = 0
etaLUT.RGBPoints = [28.799999237060547, 0.231373, 0.298039, 0.752941, 31.931509017944336, 0.865003, 0.865003, 0.865003, 35.063018798828125, 0.705882, 0.0156863, 0.14902]
etaLUT.UseLogScale = 0
etaLUT.ColorSpace = 'Diverging'
etaLUT.UseBelowRangeColor = 0
etaLUT.BelowRangeColor = [0.0, 0.0, 0.0]
etaLUT.UseAboveRangeColor = 0
etaLUT.AboveRangeColor = [0.5, 0.5, 0.5]
etaLUT.NanColor = [1.0, 1.0, 0.0]
etaLUT.NanOpacity = 1.0
etaLUT.Discretize = 1
etaLUT.NumberOfTableValues = 256
etaLUT.ScalarRangeInitialized = 1.0
etaLUT.HSVWrap = 0
etaLUT.VectorComponent = 0
etaLUT.VectorMode = 'Magnitude'
etaLUT.AllowDuplicateScalars = 1
etaLUT.Annotations = []
etaLUT.ActiveAnnotatedValues = []
etaLUT.IndexedColors = []
etaLUT.IndexedOpacities = []

# get opacity transfer function/opacity map for 'eta'
etaPWF = GetOpacityTransferFunction('eta')
etaPWF.Points = [28.799999237060547, 0.0, 0.5, 0.0, 35.063018798828125, 1.0, 0.5, 0.0]
etaPWF.AllowDuplicateScalars = 1
etaPWF.UseLogScale = 0
etaPWF.ScalarRangeInitialized = 1

# hide color bar/color legend
x_u_tst_pred_0Display.SetScalarBarVisibility(renderView1, False)

# Properties modified on renderView1
renderView1.OrientationAxesVisibility = 0

# create a new 'Plot Over Line'
plotOverLine1 = PlotOverLine(Input=x_u_tst_pred_0,
    Source='High Resolution Line Source')
plotOverLine1.PassPartialArrays = 1
plotOverLine1.ComputeTolerance = 1
plotOverLine1.Tolerance = 2.220446049250313e-16

# init the 'High Resolution Line Source' selected for 'Source'
plotOverLine1.Source.Point1 = [274670.71875, 5043520.0, 0.0]
plotOverLine1.Source.Point2 = [275234.84375, 5044085.5, 0.0]
plotOverLine1.Source.Resolution = 1000

# show data in view
plotOverLine1Display = Show(plotOverLine1, renderView1)

# trace defaults for the display properties.
plotOverLine1Display.Representation = 'Surface'
plotOverLine1Display.AmbientColor = [1.0, 1.0, 1.0]
plotOverLine1Display.ColorArrayName = ['POINTS', 'eta']
plotOverLine1Display.DiffuseColor = [1.0, 1.0, 1.0]
plotOverLine1Display.LookupTable = etaLUT
plotOverLine1Display.MapScalars = 1
plotOverLine1Display.MultiComponentsMapping = 0
plotOverLine1Display.InterpolateScalarsBeforeMapping = 1
plotOverLine1Display.Opacity = 1.0
plotOverLine1Display.PointSize = 2.0
plotOverLine1Display.LineWidth = 1.0
plotOverLine1Display.RenderLinesAsTubes = 0
plotOverLine1Display.RenderPointsAsSpheres = 0
plotOverLine1Display.Interpolation = 'Gouraud'
plotOverLine1Display.Specular = 0.0
plotOverLine1Display.SpecularColor = [1.0, 1.0, 1.0]
plotOverLine1Display.SpecularPower = 100.0
plotOverLine1Display.Luminosity = 0.0
plotOverLine1Display.Ambient = 0.0
plotOverLine1Display.Diffuse = 1.0
plotOverLine1Display.EdgeColor = [0.0, 0.0, 0.5]
plotOverLine1Display.BackfaceRepresentation = 'Follow Frontface'
plotOverLine1Display.BackfaceAmbientColor = [1.0, 1.0, 1.0]
plotOverLine1Display.BackfaceDiffuseColor = [1.0, 1.0, 1.0]
plotOverLine1Display.BackfaceOpacity = 1.0
plotOverLine1Display.Position = [0.0, 0.0, 0.0]
plotOverLine1Display.Scale = [1.0, 1.0, 1.0]
plotOverLine1Display.Orientation = [0.0, 0.0, 0.0]
plotOverLine1Display.Origin = [0.0, 0.0, 0.0]
plotOverLine1Display.Pickable = 1
plotOverLine1Display.Texture = None
plotOverLine1Display.Triangulate = 0
plotOverLine1Display.UseShaderReplacements = 0
plotOverLine1Display.ShaderReplacements = ''
plotOverLine1Display.NonlinearSubdivisionLevel = 1
plotOverLine1Display.UseDataPartitions = 0
plotOverLine1Display.OSPRayUseScaleArray = 0
plotOverLine1Display.OSPRayScaleArray = 'arc_length'
plotOverLine1Display.OSPRayScaleFunction = 'PiecewiseFunction'
plotOverLine1Display.OSPRayMaterial = 'None'
plotOverLine1Display.Orient = 0
plotOverLine1Display.OrientationMode = 'Direction'
plotOverLine1Display.SelectOrientationVectors = 'None'
plotOverLine1Display.Scaling = 0
plotOverLine1Display.ScaleMode = 'No Data Scaling Off'
plotOverLine1Display.ScaleFactor = 15.131250000000001
plotOverLine1Display.SelectScaleArray = 'None'
plotOverLine1Display.GlyphType = 'Arrow'
plotOverLine1Display.UseGlyphTable = 0
plotOverLine1Display.GlyphTableIndexArray = 'None'
plotOverLine1Display.UseCompositeGlyphTable = 0
plotOverLine1Display.UseGlyphCullingAndLOD = 0
plotOverLine1Display.LODValues = []
plotOverLine1Display.ColorByLODIndex = 0
plotOverLine1Display.GaussianRadius = 0.7565625
plotOverLine1Display.ShaderPreset = 'Sphere'
plotOverLine1Display.CustomTriangleScale = 3
plotOverLine1Display.CustomShader = """ // This custom shader code define a gaussian blur
 // Please take a look into vtkSMPointGaussianRepresentation.cxx
 // for other custom shader examples
 //VTK::Color::Impl
   float dist2 = dot(offsetVCVSOutput.xy,offsetVCVSOutput.xy);
   float gaussian = exp(-0.5*dist2);
   opacity = opacity*gaussian;
"""
plotOverLine1Display.Emissive = 0
plotOverLine1Display.ScaleByArray = 0
plotOverLine1Display.SetScaleArray = ['POINTS', 'arc_length']
plotOverLine1Display.ScaleArrayComponent = ''
plotOverLine1Display.UseScaleFunction = 1
plotOverLine1Display.ScaleTransferFunction = 'PiecewiseFunction'
plotOverLine1Display.OpacityByArray = 0
plotOverLine1Display.OpacityArray = ['POINTS', 'arc_length']
plotOverLine1Display.OpacityArrayComponent = ''
plotOverLine1Display.OpacityTransferFunction = 'PiecewiseFunction'
plotOverLine1Display.DataAxesGrid = 'GridAxesRepresentation'
plotOverLine1Display.SelectionCellLabelBold = 0
plotOverLine1Display.SelectionCellLabelColor = [0.0, 1.0, 0.0]
plotOverLine1Display.SelectionCellLabelFontFamily = 'Arial'
plotOverLine1Display.SelectionCellLabelFontFile = ''
plotOverLine1Display.SelectionCellLabelFontSize = 18
plotOverLine1Display.SelectionCellLabelItalic = 0
plotOverLine1Display.SelectionCellLabelJustification = 'Left'
plotOverLine1Display.SelectionCellLabelOpacity = 1.0
plotOverLine1Display.SelectionCellLabelShadow = 0
plotOverLine1Display.SelectionPointLabelBold = 0
plotOverLine1Display.SelectionPointLabelColor = [1.0, 1.0, 0.0]
plotOverLine1Display.SelectionPointLabelFontFamily = 'Arial'
plotOverLine1Display.SelectionPointLabelFontFile = ''
plotOverLine1Display.SelectionPointLabelFontSize = 18
plotOverLine1Display.SelectionPointLabelItalic = 0
plotOverLine1Display.SelectionPointLabelJustification = 'Left'
plotOverLine1Display.SelectionPointLabelOpacity = 1.0
plotOverLine1Display.SelectionPointLabelShadow = 0
plotOverLine1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'OSPRayScaleFunction'
plotOverLine1Display.OSPRayScaleFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
plotOverLine1Display.OSPRayScaleFunction.UseLogScale = 0

# init the 'Arrow' selected for 'GlyphType'
plotOverLine1Display.GlyphType.TipResolution = 6
plotOverLine1Display.GlyphType.TipRadius = 0.1
plotOverLine1Display.GlyphType.TipLength = 0.35
plotOverLine1Display.GlyphType.ShaftResolution = 6
plotOverLine1Display.GlyphType.ShaftRadius = 0.03
plotOverLine1Display.GlyphType.Invert = 0

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
plotOverLine1Display.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 231.4652862548828, 1.0, 0.5, 0.0]
plotOverLine1Display.ScaleTransferFunction.UseLogScale = 0

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
plotOverLine1Display.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 231.4652862548828, 1.0, 0.5, 0.0]
plotOverLine1Display.OpacityTransferFunction.UseLogScale = 0

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
plotOverLine1Display.DataAxesGrid.XTitle = 'X Axis'
plotOverLine1Display.DataAxesGrid.YTitle = 'Y Axis'
plotOverLine1Display.DataAxesGrid.ZTitle = 'Z Axis'
plotOverLine1Display.DataAxesGrid.XTitleColor = [1.0, 1.0, 1.0]
plotOverLine1Display.DataAxesGrid.XTitleFontFamily = 'Arial'
plotOverLine1Display.DataAxesGrid.XTitleFontFile = ''
plotOverLine1Display.DataAxesGrid.XTitleBold = 0
plotOverLine1Display.DataAxesGrid.XTitleItalic = 0
plotOverLine1Display.DataAxesGrid.XTitleFontSize = 12
plotOverLine1Display.DataAxesGrid.XTitleShadow = 0
plotOverLine1Display.DataAxesGrid.XTitleOpacity = 1.0
plotOverLine1Display.DataAxesGrid.YTitleColor = [1.0, 1.0, 1.0]
plotOverLine1Display.DataAxesGrid.YTitleFontFamily = 'Arial'
plotOverLine1Display.DataAxesGrid.YTitleFontFile = ''
plotOverLine1Display.DataAxesGrid.YTitleBold = 0
plotOverLine1Display.DataAxesGrid.YTitleItalic = 0
plotOverLine1Display.DataAxesGrid.YTitleFontSize = 12
plotOverLine1Display.DataAxesGrid.YTitleShadow = 0
plotOverLine1Display.DataAxesGrid.YTitleOpacity = 1.0
plotOverLine1Display.DataAxesGrid.ZTitleColor = [1.0, 1.0, 1.0]
plotOverLine1Display.DataAxesGrid.ZTitleFontFamily = 'Arial'
plotOverLine1Display.DataAxesGrid.ZTitleFontFile = ''
plotOverLine1Display.DataAxesGrid.ZTitleBold = 0
plotOverLine1Display.DataAxesGrid.ZTitleItalic = 0
plotOverLine1Display.DataAxesGrid.ZTitleFontSize = 12
plotOverLine1Display.DataAxesGrid.ZTitleShadow = 0
plotOverLine1Display.DataAxesGrid.ZTitleOpacity = 1.0
plotOverLine1Display.DataAxesGrid.FacesToRender = 63
plotOverLine1Display.DataAxesGrid.CullBackface = 0
plotOverLine1Display.DataAxesGrid.CullFrontface = 1
plotOverLine1Display.DataAxesGrid.GridColor = [1.0, 1.0, 1.0]
plotOverLine1Display.DataAxesGrid.ShowGrid = 0
plotOverLine1Display.DataAxesGrid.ShowEdges = 1
plotOverLine1Display.DataAxesGrid.ShowTicks = 1
plotOverLine1Display.DataAxesGrid.LabelUniqueEdgesOnly = 1
plotOverLine1Display.DataAxesGrid.AxesToLabel = 63
plotOverLine1Display.DataAxesGrid.XLabelColor = [1.0, 1.0, 1.0]
plotOverLine1Display.DataAxesGrid.XLabelFontFamily = 'Arial'
plotOverLine1Display.DataAxesGrid.XLabelFontFile = ''
plotOverLine1Display.DataAxesGrid.XLabelBold = 0
plotOverLine1Display.DataAxesGrid.XLabelItalic = 0
plotOverLine1Display.DataAxesGrid.XLabelFontSize = 12
plotOverLine1Display.DataAxesGrid.XLabelShadow = 0
plotOverLine1Display.DataAxesGrid.XLabelOpacity = 1.0
plotOverLine1Display.DataAxesGrid.YLabelColor = [1.0, 1.0, 1.0]
plotOverLine1Display.DataAxesGrid.YLabelFontFamily = 'Arial'
plotOverLine1Display.DataAxesGrid.YLabelFontFile = ''
plotOverLine1Display.DataAxesGrid.YLabelBold = 0
plotOverLine1Display.DataAxesGrid.YLabelItalic = 0
plotOverLine1Display.DataAxesGrid.YLabelFontSize = 12
plotOverLine1Display.DataAxesGrid.YLabelShadow = 0
plotOverLine1Display.DataAxesGrid.YLabelOpacity = 1.0
plotOverLine1Display.DataAxesGrid.ZLabelColor = [1.0, 1.0, 1.0]
plotOverLine1Display.DataAxesGrid.ZLabelFontFamily = 'Arial'
plotOverLine1Display.DataAxesGrid.ZLabelFontFile = ''
plotOverLine1Display.DataAxesGrid.ZLabelBold = 0
plotOverLine1Display.DataAxesGrid.ZLabelItalic = 0
plotOverLine1Display.DataAxesGrid.ZLabelFontSize = 12
plotOverLine1Display.DataAxesGrid.ZLabelShadow = 0
plotOverLine1Display.DataAxesGrid.ZLabelOpacity = 1.0
plotOverLine1Display.DataAxesGrid.XAxisNotation = 'Mixed'
plotOverLine1Display.DataAxesGrid.XAxisPrecision = 2
plotOverLine1Display.DataAxesGrid.XAxisUseCustomLabels = 0
plotOverLine1Display.DataAxesGrid.XAxisLabels = []
plotOverLine1Display.DataAxesGrid.YAxisNotation = 'Mixed'
plotOverLine1Display.DataAxesGrid.YAxisPrecision = 2
plotOverLine1Display.DataAxesGrid.YAxisUseCustomLabels = 0
plotOverLine1Display.DataAxesGrid.YAxisLabels = []
plotOverLine1Display.DataAxesGrid.ZAxisNotation = 'Mixed'
plotOverLine1Display.DataAxesGrid.ZAxisPrecision = 2
plotOverLine1Display.DataAxesGrid.ZAxisUseCustomLabels = 0
plotOverLine1Display.DataAxesGrid.ZAxisLabels = []
plotOverLine1Display.DataAxesGrid.UseCustomBounds = 0
plotOverLine1Display.DataAxesGrid.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
plotOverLine1Display.PolarAxes.Visibility = 0
plotOverLine1Display.PolarAxes.Translation = [0.0, 0.0, 0.0]
plotOverLine1Display.PolarAxes.Scale = [1.0, 1.0, 1.0]
plotOverLine1Display.PolarAxes.Orientation = [0.0, 0.0, 0.0]
plotOverLine1Display.PolarAxes.EnableCustomBounds = [0, 0, 0]
plotOverLine1Display.PolarAxes.CustomBounds = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]
plotOverLine1Display.PolarAxes.EnableCustomRange = 0
plotOverLine1Display.PolarAxes.CustomRange = [0.0, 1.0]
plotOverLine1Display.PolarAxes.PolarAxisVisibility = 1
plotOverLine1Display.PolarAxes.RadialAxesVisibility = 1
plotOverLine1Display.PolarAxes.DrawRadialGridlines = 1
plotOverLine1Display.PolarAxes.PolarArcsVisibility = 1
plotOverLine1Display.PolarAxes.DrawPolarArcsGridlines = 1
plotOverLine1Display.PolarAxes.NumberOfRadialAxes = 0
plotOverLine1Display.PolarAxes.AutoSubdividePolarAxis = 1
plotOverLine1Display.PolarAxes.NumberOfPolarAxis = 0
plotOverLine1Display.PolarAxes.MinimumRadius = 0.0
plotOverLine1Display.PolarAxes.MinimumAngle = 0.0
plotOverLine1Display.PolarAxes.MaximumAngle = 90.0
plotOverLine1Display.PolarAxes.RadialAxesOriginToPolarAxis = 1
plotOverLine1Display.PolarAxes.Ratio = 1.0
plotOverLine1Display.PolarAxes.PolarAxisColor = [1.0, 1.0, 1.0]
plotOverLine1Display.PolarAxes.PolarArcsColor = [1.0, 1.0, 1.0]
plotOverLine1Display.PolarAxes.LastRadialAxisColor = [1.0, 1.0, 1.0]
plotOverLine1Display.PolarAxes.SecondaryPolarArcsColor = [1.0, 1.0, 1.0]
plotOverLine1Display.PolarAxes.SecondaryRadialAxesColor = [1.0, 1.0, 1.0]
plotOverLine1Display.PolarAxes.PolarAxisTitleVisibility = 1
plotOverLine1Display.PolarAxes.PolarAxisTitle = 'Radial Distance'
plotOverLine1Display.PolarAxes.PolarAxisTitleLocation = 'Bottom'
plotOverLine1Display.PolarAxes.PolarLabelVisibility = 1
plotOverLine1Display.PolarAxes.PolarLabelFormat = '%-#6.3g'
plotOverLine1Display.PolarAxes.PolarLabelExponentLocation = 'Labels'
plotOverLine1Display.PolarAxes.RadialLabelVisibility = 1
plotOverLine1Display.PolarAxes.RadialLabelFormat = '%-#3.1f'
plotOverLine1Display.PolarAxes.RadialLabelLocation = 'Bottom'
plotOverLine1Display.PolarAxes.RadialUnitsVisibility = 1
plotOverLine1Display.PolarAxes.ScreenSize = 10.0
plotOverLine1Display.PolarAxes.PolarAxisTitleColor = [1.0, 1.0, 1.0]
plotOverLine1Display.PolarAxes.PolarAxisTitleOpacity = 1.0
plotOverLine1Display.PolarAxes.PolarAxisTitleFontFamily = 'Arial'
plotOverLine1Display.PolarAxes.PolarAxisTitleFontFile = ''
plotOverLine1Display.PolarAxes.PolarAxisTitleBold = 0
plotOverLine1Display.PolarAxes.PolarAxisTitleItalic = 0
plotOverLine1Display.PolarAxes.PolarAxisTitleShadow = 0
plotOverLine1Display.PolarAxes.PolarAxisTitleFontSize = 12
plotOverLine1Display.PolarAxes.PolarAxisLabelColor = [1.0, 1.0, 1.0]
plotOverLine1Display.PolarAxes.PolarAxisLabelOpacity = 1.0
plotOverLine1Display.PolarAxes.PolarAxisLabelFontFamily = 'Arial'
plotOverLine1Display.PolarAxes.PolarAxisLabelFontFile = ''
plotOverLine1Display.PolarAxes.PolarAxisLabelBold = 0
plotOverLine1Display.PolarAxes.PolarAxisLabelItalic = 0
plotOverLine1Display.PolarAxes.PolarAxisLabelShadow = 0
plotOverLine1Display.PolarAxes.PolarAxisLabelFontSize = 12
plotOverLine1Display.PolarAxes.LastRadialAxisTextColor = [1.0, 1.0, 1.0]
plotOverLine1Display.PolarAxes.LastRadialAxisTextOpacity = 1.0
plotOverLine1Display.PolarAxes.LastRadialAxisTextFontFamily = 'Arial'
plotOverLine1Display.PolarAxes.LastRadialAxisTextFontFile = ''
plotOverLine1Display.PolarAxes.LastRadialAxisTextBold = 0
plotOverLine1Display.PolarAxes.LastRadialAxisTextItalic = 0
plotOverLine1Display.PolarAxes.LastRadialAxisTextShadow = 0
plotOverLine1Display.PolarAxes.LastRadialAxisTextFontSize = 12
plotOverLine1Display.PolarAxes.SecondaryRadialAxesTextColor = [1.0, 1.0, 1.0]
plotOverLine1Display.PolarAxes.SecondaryRadialAxesTextOpacity = 1.0
plotOverLine1Display.PolarAxes.SecondaryRadialAxesTextFontFamily = 'Arial'
plotOverLine1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''
plotOverLine1Display.PolarAxes.SecondaryRadialAxesTextBold = 0
plotOverLine1Display.PolarAxes.SecondaryRadialAxesTextItalic = 0
plotOverLine1Display.PolarAxes.SecondaryRadialAxesTextShadow = 0
plotOverLine1Display.PolarAxes.SecondaryRadialAxesTextFontSize = 12
plotOverLine1Display.PolarAxes.EnableDistanceLOD = 1
plotOverLine1Display.PolarAxes.DistanceLODThreshold = 0.7
plotOverLine1Display.PolarAxes.EnableViewAngleLOD = 1
plotOverLine1Display.PolarAxes.ViewAngleLODThreshold = 0.7
plotOverLine1Display.PolarAxes.SmallestVisiblePolarAngle = 0.5
plotOverLine1Display.PolarAxes.PolarTicksVisibility = 1
plotOverLine1Display.PolarAxes.ArcTicksOriginToPolarAxis = 1
plotOverLine1Display.PolarAxes.TickLocation = 'Both'
plotOverLine1Display.PolarAxes.AxisTickVisibility = 1
plotOverLine1Display.PolarAxes.AxisMinorTickVisibility = 0
plotOverLine1Display.PolarAxes.ArcTickVisibility = 1
plotOverLine1Display.PolarAxes.ArcMinorTickVisibility = 0
plotOverLine1Display.PolarAxes.DeltaAngleMajor = 10.0
plotOverLine1Display.PolarAxes.DeltaAngleMinor = 5.0
plotOverLine1Display.PolarAxes.PolarAxisMajorTickSize = 0.0
plotOverLine1Display.PolarAxes.PolarAxisTickRatioSize = 0.3
plotOverLine1Display.PolarAxes.PolarAxisMajorTickThickness = 1.0
plotOverLine1Display.PolarAxes.PolarAxisTickRatioThickness = 0.5
plotOverLine1Display.PolarAxes.LastRadialAxisMajorTickSize = 0.0
plotOverLine1Display.PolarAxes.LastRadialAxisTickRatioSize = 0.3
plotOverLine1Display.PolarAxes.LastRadialAxisMajorTickThickness = 1.0
plotOverLine1Display.PolarAxes.LastRadialAxisTickRatioThickness = 0.5
plotOverLine1Display.PolarAxes.ArcMajorTickSize = 0.0
plotOverLine1Display.PolarAxes.ArcTickRatioSize = 0.3
plotOverLine1Display.PolarAxes.ArcMajorTickThickness = 1.0
plotOverLine1Display.PolarAxes.ArcTickRatioThickness = 0.5
plotOverLine1Display.PolarAxes.Use2DMode = 0
plotOverLine1Display.PolarAxes.UseLogAxis = 0

# Create a new 'Line Chart View'
lineChartView1 = CreateView('XYChartView')
lineChartView1.UseCache = 0
lineChartView1.ViewSize = [400, 400]
lineChartView1.ChartTitle = ''
lineChartView1.ChartTitleAlignment = 'Center'
lineChartView1.ChartTitleFontFamily = 'Arial'
lineChartView1.ChartTitleFontFile = ''
lineChartView1.ChartTitleFontSize = 18
lineChartView1.ChartTitleBold = 0
lineChartView1.ChartTitleItalic = 0
lineChartView1.ChartTitleColor = [0.0, 0.0, 0.0]
lineChartView1.ShowLegend = 1
lineChartView1.LegendLocation = 'TopRight'
lineChartView1.SortByXAxis = 0
lineChartView1.LegendPosition = [0, 0]
lineChartView1.LegendFontFamily = 'Arial'
lineChartView1.LegendFontFile = ''
lineChartView1.LegendFontSize = 12
lineChartView1.LegendBold = 0
lineChartView1.LegendItalic = 0
lineChartView1.TooltipNotation = 'Mixed'
lineChartView1.TooltipPrecision = 6
lineChartView1.HideTimeMarker = 0
lineChartView1.LeftAxisTitle = ''
lineChartView1.ShowLeftAxisGrid = 1
lineChartView1.LeftAxisGridColor = [0.95, 0.95, 0.95]
lineChartView1.LeftAxisColor = [0.0, 0.0, 0.0]
lineChartView1.LeftAxisTitleFontFamily = 'Arial'
lineChartView1.LeftAxisTitleFontFile = ''
lineChartView1.LeftAxisTitleFontSize = 18
lineChartView1.LeftAxisTitleBold = 1
lineChartView1.LeftAxisTitleItalic = 0
lineChartView1.LeftAxisTitleColor = [0.0, 0.0, 0.0]
lineChartView1.LeftAxisLogScale = 0
lineChartView1.LeftAxisUseCustomRange = 0
lineChartView1.LeftAxisRangeMinimum = 0.0
lineChartView1.LeftAxisRangeMaximum = 1.0
lineChartView1.ShowLeftAxisLabels = 1
lineChartView1.LeftAxisLabelNotation = 'Mixed'
lineChartView1.LeftAxisLabelPrecision = 2
lineChartView1.LeftAxisUseCustomLabels = 0
lineChartView1.LeftAxisLabels = []
lineChartView1.LeftAxisLabelFontFamily = 'Arial'
lineChartView1.LeftAxisLabelFontFile = ''
lineChartView1.LeftAxisLabelFontSize = 12
lineChartView1.LeftAxisLabelBold = 0
lineChartView1.LeftAxisLabelItalic = 0
lineChartView1.LeftAxisLabelColor = [0.0, 0.0, 0.0]
lineChartView1.BottomAxisTitle = ''
lineChartView1.ShowBottomAxisGrid = 1
lineChartView1.BottomAxisGridColor = [0.95, 0.95, 0.95]
lineChartView1.BottomAxisColor = [0.0, 0.0, 0.0]
lineChartView1.BottomAxisTitleFontFamily = 'Arial'
lineChartView1.BottomAxisTitleFontFile = ''
lineChartView1.BottomAxisTitleFontSize = 18
lineChartView1.BottomAxisTitleBold = 1
lineChartView1.BottomAxisTitleItalic = 0
lineChartView1.BottomAxisTitleColor = [0.0, 0.0, 0.0]
lineChartView1.BottomAxisLogScale = 0
lineChartView1.BottomAxisUseCustomRange = 0
lineChartView1.BottomAxisRangeMinimum = 0.0
lineChartView1.BottomAxisRangeMaximum = 1.0
lineChartView1.ShowBottomAxisLabels = 1
lineChartView1.BottomAxisLabelNotation = 'Mixed'
lineChartView1.BottomAxisLabelPrecision = 2
lineChartView1.BottomAxisUseCustomLabels = 0
lineChartView1.BottomAxisLabels = []
lineChartView1.BottomAxisLabelFontFamily = 'Arial'
lineChartView1.BottomAxisLabelFontFile = ''
lineChartView1.BottomAxisLabelFontSize = 12
lineChartView1.BottomAxisLabelBold = 0
lineChartView1.BottomAxisLabelItalic = 0
lineChartView1.BottomAxisLabelColor = [0.0, 0.0, 0.0]
lineChartView1.RightAxisTitle = ''
lineChartView1.ShowRightAxisGrid = 1
lineChartView1.RightAxisGridColor = [0.95, 0.95, 0.95]
lineChartView1.RightAxisColor = [0.0, 0.0, 0.0]
lineChartView1.RightAxisTitleFontFamily = 'Arial'
lineChartView1.RightAxisTitleFontFile = ''
lineChartView1.RightAxisTitleFontSize = 18
lineChartView1.RightAxisTitleBold = 1
lineChartView1.RightAxisTitleItalic = 0
lineChartView1.RightAxisTitleColor = [0.0, 0.0, 0.0]
lineChartView1.RightAxisLogScale = 0
lineChartView1.RightAxisUseCustomRange = 0
lineChartView1.RightAxisRangeMinimum = 0.0
lineChartView1.RightAxisRangeMaximum = 1.0
lineChartView1.ShowRightAxisLabels = 1
lineChartView1.RightAxisLabelNotation = 'Mixed'
lineChartView1.RightAxisLabelPrecision = 2
lineChartView1.RightAxisUseCustomLabels = 0
lineChartView1.RightAxisLabels = []
lineChartView1.RightAxisLabelFontFamily = 'Arial'
lineChartView1.RightAxisLabelFontFile = ''
lineChartView1.RightAxisLabelFontSize = 12
lineChartView1.RightAxisLabelBold = 0
lineChartView1.RightAxisLabelItalic = 0
lineChartView1.RightAxisLabelColor = [0.0, 0.0, 0.0]
lineChartView1.TopAxisTitle = ''
lineChartView1.ShowTopAxisGrid = 1
lineChartView1.TopAxisGridColor = [0.95, 0.95, 0.95]
lineChartView1.TopAxisColor = [0.0, 0.0, 0.0]
lineChartView1.TopAxisTitleFontFamily = 'Arial'
lineChartView1.TopAxisTitleFontFile = ''
lineChartView1.TopAxisTitleFontSize = 18
lineChartView1.TopAxisTitleBold = 1
lineChartView1.TopAxisTitleItalic = 0
lineChartView1.TopAxisTitleColor = [0.0, 0.0, 0.0]
lineChartView1.TopAxisLogScale = 0
lineChartView1.TopAxisUseCustomRange = 0
lineChartView1.TopAxisRangeMinimum = 0.0
lineChartView1.TopAxisRangeMaximum = 1.0
lineChartView1.ShowTopAxisLabels = 1
lineChartView1.TopAxisLabelNotation = 'Mixed'
lineChartView1.TopAxisLabelPrecision = 2
lineChartView1.TopAxisUseCustomLabels = 0
lineChartView1.TopAxisLabels = []
lineChartView1.TopAxisLabelFontFamily = 'Arial'
lineChartView1.TopAxisLabelFontFile = ''
lineChartView1.TopAxisLabelFontSize = 12
lineChartView1.TopAxisLabelBold = 0
lineChartView1.TopAxisLabelItalic = 0
lineChartView1.TopAxisLabelColor = [0.0, 0.0, 0.0]

# show data in view
plotOverLine1Display_1 = Show(plotOverLine1, lineChartView1)

# trace defaults for the display properties.
plotOverLine1Display_1.CompositeDataSetIndex = [0]
plotOverLine1Display_1.AttributeType = 'Point Data'
plotOverLine1Display_1.UseIndexForXAxis = 0
plotOverLine1Display_1.XArrayName = 'arc_length'
plotOverLine1Display_1.SeriesVisibility = ['eta', 'eta_pred', 'eta_pred_lo', 'eta_pred_up']
plotOverLine1Display_1.SeriesLabel = ['arc_length', 'arc_length', 'eta', 'eta', 'eta_pred', 'eta_pred', 'eta_pred_lo', 'eta_pred_lo', 'eta_pred_up', 'eta_pred_up', 'vtkValidPointMask', 'vtkValidPointMask', 'Points_X', 'Points_X', 'Points_Y', 'Points_Y', 'Points_Z', 'Points_Z', 'Points_Magnitude', 'Points_Magnitude']
plotOverLine1Display_1.SeriesColor = ['arc_length', '0', '0', '0', 'eta', '0.89', '0.1', '0.11', 'eta_pred', '0.22', '0.49', '0.72', 'eta_pred_lo', '0.3', '0.69', '0.29', 'eta_pred_up', '0.6', '0.31', '0.64', 'vtkValidPointMask', '1', '0.5', '0', 'Points_X', '0.65', '0.34', '0.16', 'Points_Y', '0', '0', '0', 'Points_Z', '0.89', '0.1', '0.11', 'Points_Magnitude', '0.22', '0.49', '0.72']
plotOverLine1Display_1.SeriesPlotCorner = ['arc_length', '0', 'eta', '0', 'eta_pred', '0', 'eta_pred_lo', '0', 'eta_pred_up', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']
plotOverLine1Display_1.SeriesLabelPrefix = ''
plotOverLine1Display_1.SeriesLineStyle = ['arc_length', '1', 'eta', '1', 'eta_pred', '1', 'eta_pred_lo', '1', 'eta_pred_up', '1', 'vtkValidPointMask', '1', 'Points_X', '1', 'Points_Y', '1', 'Points_Z', '1', 'Points_Magnitude', '1']
plotOverLine1Display_1.SeriesLineThickness = ['arc_length', '2', 'eta', '2', 'eta_pred', '2', 'eta_pred_lo', '2', 'eta_pred_up', '2', 'vtkValidPointMask', '2', 'Points_X', '2', 'Points_Y', '2', 'Points_Z', '2', 'Points_Magnitude', '2']
plotOverLine1Display_1.SeriesMarkerStyle = ['arc_length', '0', 'eta', '0', 'eta_pred', '0', 'eta_pred_lo', '0', 'eta_pred_up', '0', 'vtkValidPointMask', '0', 'Points_X', '0', 'Points_Y', '0', 'Points_Z', '0', 'Points_Magnitude', '0']

# add view to a layout so it's visible in UI
AssignViewToLayout(view=lineChartView1, layout=layout1, hint=3)

# update the view to ensure updated data information
renderView1.Update()

# update the view to ensure updated data information
lineChartView1.Update()

# Rescale transfer function
etaLUT.RescaleTransferFunction(28.7999992371, 35.0630187988)

# Rescale transfer function
etaPWF.RescaleTransferFunction(28.7999992371, 35.0630187988)

# set active view
SetActiveView(renderView1)

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [274952.78125, 5043802.75, 10000.0]
renderView1.CameraFocalPoint = [274952.78125, 5043802.75, 0.0]
renderView1.CameraParallelScale = 272.7839933853791

# save screenshot
SaveScreenshot('/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred.0.png', renderView1, SaveAllViews=0,
    ImageResolution=[162, 393],
    FontScaling='Scale fonts proportionally',
    SeparatorWidth=1,
    SeparatorColor=[0.937, 0.922, 0.906],
    OverrideColorPalette='',
    StereoMode='No change',
    TransparentBackground=0, 
    # PNG options
    CompressionLevel='5')

# Properties modified on animationScene1
animationScene1.AnimationTime = 1.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 1.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 2.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 2.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 3.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 3.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 4.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 4.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 5.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 5.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 4.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 4.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 5.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 5.0

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [274952.78125, 5043802.75, 10000.0]
renderView1.CameraFocalPoint = [274952.78125, 5043802.75, 0.0]
renderView1.CameraParallelScale = 272.7839933853791

# save screenshot
SaveScreenshot('/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred.5.png', renderView1, SaveAllViews=0,
    ImageResolution=[162, 393],
    FontScaling='Scale fonts proportionally',
    SeparatorWidth=1,
    SeparatorColor=[0.937, 0.922, 0.906],
    OverrideColorPalette='',
    StereoMode='No change',
    TransparentBackground=0, 
    # PNG options
    CompressionLevel='5')

# Properties modified on animationScene1
animationScene1.AnimationTime = 6.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 6.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 7.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 7.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 8.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 8.0

# Properties modified on animationScene1
animationScene1.AnimationTime = 9.0

# Properties modified on timeKeeper1
timeKeeper1.Time = 9.0

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [274952.78125, 5043802.75, 10000.0]
renderView1.CameraFocalPoint = [274952.78125, 5043802.75, 0.0]
renderView1.CameraParallelScale = 272.7839933853791

# save screenshot
SaveScreenshot('/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred.9.png', renderView1, SaveAllViews=0,
    ImageResolution=[162, 393],
    FontScaling='Scale fonts proportionally',
    SeparatorWidth=1,
    SeparatorColor=[0.937, 0.922, 0.906],
    OverrideColorPalette='',
    StereoMode='No change',
    TransparentBackground=0, 
    # PNG options
    CompressionLevel='5')

# save data
SaveData('/Users/pierremtb/LocalWork/ETS/POD-EnsNN/examples/2dt_shallowwater/cache/x_u_tst_pred.csv', proxy=plotOverLine1, WriteTimeSteps=1,
    Filenamesuffix='.%d',
    Precision=5,
    FieldDelimiter=',',
    UseScientificNotation=0,
    FieldAssociation='Points',
    AddMetaData=1)

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.InteractionMode = '2D'
renderView1.CameraPosition = [274952.78125, 5043802.75, 10000.0]
renderView1.CameraFocalPoint = [274952.78125, 5043802.75, 0.0]
renderView1.CameraParallelScale = 272.7839933853791

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).
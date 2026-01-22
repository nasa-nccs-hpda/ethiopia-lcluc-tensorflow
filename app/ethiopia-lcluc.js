///////////////////////////////////////////////////////////////
  //                    1) Import Layers of Interest           //
  ///////////////////////////////////////////////////////////////
  
  // set boundary
  var ethiopiaBoundary = ee.FeatureCollection('projects/ee-jacaraba-ethiopia/assets/boundaries/Amhara_Study_Area_Boundary_4buf10km_EPSG_GEE')
  
  // data products
  var worldCov10m = ee.ImageCollection('ESA/WorldCover/v100').first();
  var gladCropland = ee.ImageCollection('users/potapovpeter/Global_cropland_2019');
  var alemuLandCover = ee.ImageCollection('ESA/WorldCover/v100').first();//ee.Image('projects/ee-jacaraba-ethiopia/assets/composites/Amhara_M1BS_2016-2024_mode')
  var canopyHeight = ee.ImageCollection("projects/sat-io/open-datasets/facebook/meta-canopy-height").mosaic();
  
  // crop worldcov to the region of interest
  worldCov10m = worldCov10m.clip(ethiopiaBoundary)
  
  // glad cropland clip
  gladCropland = gladCropland.map(function(image){
    return image.clip(ethiopiaBoundary)
  })
  
  // mask cropland
  var gladCroplandMasked = gladCropland.map(function(image){
    var masked = image.gt(0)
    return image.updateMask(masked)
  })
  
  // crop Alemu's map to the region of interest
  alemuLandCover = alemuLandCover.clip(ethiopiaBoundary)
  
  canopyHeight = canopyHeight.clip(ethiopiaBoundary)
  
  ///////////////////////////////////////////////////////////////
  //      2) Begin setting up map appearance and app layers   //
  ///////////////////////////////////////////////////////////////
  
  // Set up a satellite background
  Map.setOptions('Satellite')
  
  // Center the map to Guyana
  Map.centerObject(ethiopiaBoundary, 7)
  
  // Change style of cursor to 'crosshair'
  Map.style().set('cursor', 'crosshair');
  
  // Add WorldCov
  var visualizationWorldCov10m = {
    bands: ['Map'],
  };
  
  //// viz settings //// 
  var palettes = require('users/gena/packages:palettes');
  
  var worldCov10mLayer = ui.Map.Layer(worldCov10m, visualizationWorldCov10m, 'WorldCov 10m', false)
  var gladCroplandLayer = ui.Map.Layer(gladCroplandMasked, {palette:['#FFA500'], min:0, max:1}, 'Glad Cropland', false)
  var alemuLandCoverLayer = ui.Map.Layer(alemuLandCover, {palette:['#FFA500'], min:0, max:1}, 'Alemu LC 2m', false)
  var metaCHMLayer = ui.Map.Layer(canopyHeight, {palette: palettes.matplotlib.viridis[7], min:0, max:20}, 'Meta CHM', false)
  
  Map.add(worldCov10mLayer);
  Map.add(gladCroplandLayer);
  Map.add(alemuLandCoverLayer);
  Map.add(metaCHMLayer);
  
  ///////////////////////////////////////////////////////////////
  //      3) Set up panels and widgets for display             //
  ///////////////////////////////////////////////////////////////
  
  //3.1) Set up title and summary widgets
  
  //App title
  var header = ui.Label('Ethiopia Land Cover Extent Explorer', {fontSize: '25px', fontWeight: 'bold', color: '4A997E'});
  
  //App summary
  var text = ui.Label(
    'This is work in progress. This tool compares several land cover products in Ethiopia ' +
    'at different spatial resolutions. Use the tools below to explore land cover extents.',
      {fontSize: '15px'});
  
  
  //3.2) Create a panel to hold text
  var panel = ui.Panel({
    widgets:[header, text],//Adds header and text
    style:{width: '300px', position:'middle-right'}});
  
  
  //3.3) Create variable for additional text and separators
  
  //This creates another panel to house a line separator and instructions for the user
  var intro = ui.Panel([
    ui.Label({
      value: '____________________________________________',
      style: {fontWeight: 'bold',  color: '4A997E'},
    }),
    ui.Label({
      value:'Select layers to display.',
      style: {fontSize: '15px', fontWeight: 'bold'}
    })]);
  
  //Add this new panel to the larger panel we created 
  panel.add(intro)
  
  //3.4) Add our main panel to the root of our GUI
  ui.root.insert(1, panel)
  
  
  ///////////////////////////////////////////////////////////////
  //         4) Add checkbox widgets and legends               //
  ///////////////////////////////////////////////////////////////
  
  //4.1) Create a new label for this series of checkboxes
  
  var extLabel = ui.Label({value:'Land Cover Extent',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '10px 5px'}
  });
  
  //4.2) Add checkboxes to our display
  
  //Create checkboxes that will allow the user to view the extent map for different years
  //Creating the checkbox will not do anything yet, we add functionality further 
  // in the code
  
  var extCheck = ui.Checkbox('WorldCov 10m').setValue(false); //false = unchecked
  
  var extCheck2 = ui.Checkbox('Glad Cropland').setValue(false);
  
  var extCheck3 = ui.Checkbox('Alemu LC 2m').setValue(false);
  
  var extCheck4 = ui.Checkbox('Meta CHM 1m').setValue(false);
  
  
  //Extent Legend
  ///////////////
  
  // Set position of panel
  var extentLegend = ui.Panel({
    style: {
      position: 'bottom-left',
      padding: '8px 15px'
    }
  });
  
  var extentLegendGlad = ui.Panel({
    style: {
      position: 'bottom-left',
      padding: '8px 15px'
    }
  });
  
  var extentLegendAlemu = ui.Panel({
    style: {
      position: 'bottom-left',
      padding: '8px 15px'
    }
  });
  
  // The following creates and styles 1 row of the legend.
  var makeRowa = function(color, name) {
   
        // Create the label that is actually the colored box.
        var colorBox = ui.Label({
          style: {
            backgroundColor: '#' + color,
            // Use padding to give the box height and width.
            padding: '8px',
            margin: '0 0 4px 0'
          }
        });
   
        // Create a label with the description text.
        var description = ui.Label({
          value: name,
          style: {margin: '0 0 4px 6px'}
        });
   
        // Return the panel
        return ui.Panel({
          widgets: [colorBox, description],
          layout: ui.Panel.Layout.Flow('horizontal')
        });
  };
  
  
  //Create a palette using the same colors we used for each extent layer
  var paletteMAPa = [
    "006400",
    "ffbb22",
    "ffff4c",
    "f096ff",
    "fa0000",
    "b4b4b4",
    "f0f0f0",
    "0064c8",
    "0096a0",
    "00cf75",
    "fae6a0",
  ];
  
  
  // Name of each legend value
  var namesa = [
    "10 Trees", "20 Shrubland", "30 Grassland", 
    "40 Cropland", "50 Built-up", "60 Barren / sparse vegetation",
    "70 Snow and ice", "80 Open water", "90 Herbaceous wetland",
    "95 Mangroves", "100 Moss and lichen"
  ]; 
             
   
  // Add color and names to legend
  for (var i = 0; i < paletteMAPa.length; i++) {
    extentLegend.add(makeRowa(paletteMAPa[i], namesa[i]));
    }  
  
  // Add color and names to legend
  for (var i = 0; i < paletteMAPa.length; i++) {
    extentLegendAlemu.add(makeRowa(paletteMAPa[i], namesa[i]));
    }  
    
  extentLegendGlad.add(makeRowa('ffa500', 'Cropland'));
  
  
  //Height Legend
  ///////////////
  
  // This uses function to construct a legend for the given single-band vis
  // parameters.  Requires that the vis parameters specify 'min' and 
  // 'max' but not 'bands'.
  function makeLegend2 (viridis) {
    var lon = ee.Image.pixelLonLat().select('longitude');
    var gradient = lon.multiply((viridis.max-viridis.min)/100.0).add(viridis.min);
    var legendImage = gradient.visualize(viridis);
    
    var thumb = ui.Thumbnail({
      image: legendImage, 
      params: {bbox:'0,0,100,8', dimensions:'256x20'},  
      style: {position: 'bottom-center'}
    });
    var panel2 = ui.Panel({
      widgets: [
        ui.Label('5 m'), 
        ui.Label({style: {stretch: 'horizontal'}}), 
        ui.Label('45 m')
      ],
      layout: ui.Panel.Layout.flow('horizontal'),
      style: {stretch: 'horizontal', maxWidth: '270px', padding: '0px 0px 0px 8px'}
    });
    return ui.Panel().add(panel2).add(thumb);
  }
  
  var extLabelWorldCovLegend = ui.Label({value:'World Cov Legend',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '10px 5px'}
  });
  
  var extLabelGladLegend = ui.Label({value:'GLAD Cropland Legend',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '10px 5px'}
  });
  
  var extLabelAlemuLegend = ui.Label({value:'Alemu Land Cover Legend',
  style: {fontWeight: 'bold', fontSize: '16px', margin: '10px 5px'}
  });
  
  //4.4) Add these new widgets to the panel in the order you want them to appear
  panel.add(extLabel)
        .add(extCheck)
        .add(extCheck2)
        .add(extCheck3)
        .add(extCheck4)
        .add(extLabelWorldCovLegend)
        .add(extentLegend)
        .add(extLabelGladLegend)
        .add(extentLegendGlad)
        .add(extLabelAlemuLegend)
        .add(extentLegendAlemu)
        
        
  //Extent 2000
  var doCheckbox = function() {
    
    extCheck.onChange(function(checked){
      worldCov10mLayer.setShown(checked)
    })
  }
  doCheckbox();
  
  //Extent 2010
  var doCheckbox2 = function() {
    
    extCheck2.onChange(function(checked){
      gladCroplandLayer.setShown(checked)
    })
    
  
  }
  doCheckbox2();
  
  
  //Extent 2010
  var doCheckbox3 = function() {
    
    extCheck3.onChange(function(checked){
      alemuLandCoverLayer.setShown(checked)
    })
    
  
  }
  doCheckbox3();
  
  var doCheckbox4 = function() {
    
    extCheck4.onChange(function(checked){
      metaCHMLayer.setShown(checked)
    })
    
  
  }
  doCheckbox4();

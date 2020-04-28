# PROJ0016 - Big Data Project - Renewable Energy Production Forecast

How much renewable energy will we produce tomorrow in Liège ?

- Yann Claes
- Gaspard Lambrechts
- François Rozet

November 28, 2019

## Scope of the project

In order to predict the amount of renewable energy produced in the municipality of Liège (tomorrow), we have chosen to decompose the problem in several easier sub-problems and to bind their respective solution into a global answer later on.

### Production models

The first part would be build production models for both wind turbines and photovoltaic panels.

Initially, we could make use of theoretical models to predict their electrical production :

- Transfer function of wind turbines according to the wind (and brand/model);
- Production model of photovoltaic panels with respect to the irradiance, temperature and orientation (and brand/model).

However, it would be interesting to verify the accuracy of these theoretical models on known data. For example, on the photovoltaic panels of the Grands Amphitéatres’ parkings (cf. [Acquired data](#Acquired-data)). We also want to build empirical models using supervised learning on these data.

### Production units

Secondly, we will gather information about the number and position of units of photovoltaic panels and wind turbines in the municipality of Liège.

We already possess (cf. [Acquired data](#Acquired-data)) such data. However, two things could motivate another approach for the census of photovoltaic panels:

- The data obtained by the [CWaPE][CWaPE] is two years old and gives no information on the location, orientation, tilt and model/type of the photovoltaic panels within each municipality.
- If we only rely on existing sources for the census, as soon as no data is available, we are not able to forecast anything.

Thus, it has been chosen (for now) to estimate the number and position of photovoltaic panels in the considered area by using satellite imagery. The strength of this approach is that it is not restricted to any area or scale.

### Global production forecast

Thirdly, combining the production models and production units data, we should be able to predict the global renewable energy production in Liège, but also in Walloonia and Belgium with respect to the *weather forecast*.

Finally, in order to assess the quality of our predictions, we will compare our model with the national/regional production data from [Elia][Elia]/[APERe][APERe].

However, it should be noted that the production measures of [Elia][Elia] and [APERe][APERe] probably don't take into account the electrical production which is not *re-injected* in the network, i.e. the part that is directly consumed by the owners.

## Models

### Production models

- Photovoltaic panels
    - Using the theoretical model;
    - Using the [`pvlib`][pvlib] Python library;
    - Testing the model on the production data from the PV of the *Grands Amphitéatres*' parkings and the weather data from the [Laboratoire de Climatologie de l'Uliège][Climato] weather station.
- Wind farms (using the theoretical model)
    - Using the theoretical model.
- Others for any other source of production

### Production units number estimation

- Photovoltaic panels (Computer vision and deep learning from satellite imagery)
    - [DeepSolar][DeepSolar] could be an inspiration
- Wind farms 
    - Data collection from each electricity provider
- Other renewable sources of production (to be investigated)

## Acquired data

[acquired-no]: https://img.shields.io/badge/Acquired-No-Red.svg
[acquired-yes]: https://img.shields.io/badge/Acquired-Yes-Green.svg
[acquired-asked]: https://img.shields.io/badge/Acquired-Asked-Orange.svg

### Weather data and weather forecast

- [Laboratoire de Climatologie de l'Uliège][Climato]
    - Measures near B11 building, Sart-Tilman
        - Solar flow and forecast (May - October 2019) ![][acquired-yes]
        -  Temperatures and forecast (May - October 2019) ![][acquired-yes]
    - Measures at Sart-Tilman Sud
        - Solar flow, temperatures and forecast (August - November 2019) ![][acquired-yes]
- Weather data from the [Thermodynamics Laboratory from the Aerospace and Mechanical Engineering Department of Liège University][Thermo]  ![][acquired-yes]
- Public API's [OpenWeatherMap][OpenWeatherMap] ![][acquired-no]

### Production of photovoltaic panels

- Production of the photovoltaic panels of the *Grands Amphitéatres*' parkings (MySQL access to real time and past data) ![][acquired-yes]

### Number and position of wind turbine

- All recorded wind farms from [Elia][Elia] ![][acquired-yes]
- Data from each electricity supplier having wind turbines:
    - List of electricity suppliers ![][acquired-yes]
    - Data from each supplier ![][acquired-no]

### Number and position of photovoltaic panels

- Number of photovoltaic panels and installed power per municipality [![acquired-yes]](https://www.cwape.be/docs/?doc=1529)
    
    > Two years old, no data on the orientation and tilt 

### National and regional renewable energy production

- PV power every 15 minutes from [Elia][Elia] ![][acquired-asked] or [APERe][APERe] ![][acquired-asked]
- Wind power every 15 minutes from [Elia][Elia] ![][acquired-asked] or [APERe][APERe] ![][acquired-asked] 

[Elia]: https://www.elia.be/
[APERe]: http://www.apere.org/
[CWaPE]: https://www.cwape.be/
[Climato]: http://climato.be/cms/index.php?climato=releves-meteorologique-au-sart-tilman
[Thermo]: http://www.labothap.ulg.ac.be/cmsms/
[OpenWeatherMap]: https://openweathermap.org/
[DeepSolar]: https://github.com/wangzhecheng/DeepSolar
[pvlib]: https://github.com/pvlib/pvlib-python

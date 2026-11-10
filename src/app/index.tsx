// Imports
import {
  View,
  Text,
  TouchableOpacity,
  StyleSheet,
  Image,
} from 'react-native';

import React, { useState } from 'react';
import { SafeAreaView } from 'react-native-safe-area-context';

import CameraScreen from './screens/CameraScreen';
import GalleryScreen from './screens/GalleryScreen';
import SnakeEncyclopediaScreen from './screens/SnakeEncyclopediaScreen';


// Main
export default function Index() {
  const [currentPage, setCurrentPage] = useState('Home');

  return (
    <SafeAreaView style={styles.container} edges={['bottom']}>

      {/* Pages */}
      <View style={styles.content}>
        {currentPage === 'Home' && <CameraScreen />}
        {currentPage === 'Gallery' && <GalleryScreen />}
        {currentPage === 'SnakeEncyclopedia' && <SnakeEncyclopediaScreen />}
      </View>

      {/* Navigation Tab */}
      <View style={styles.navigation}>

        <TouchableOpacity onPress={() => setCurrentPage('Gallery')}>
          <Image
            source={require('@/assets/images/photolibrary.png')}
            style={styles.icon}
            resizeMode="contain"
          />
        </TouchableOpacity>

        <TouchableOpacity onPress={() => setCurrentPage('Home')}>
          <Image
            source={require('@/assets/images/photocamera.png')}
            style={styles.icon}
            resizeMode="contain"
          />
        </TouchableOpacity>

        <TouchableOpacity onPress={() => setCurrentPage('SnakeEncyclopedia')}>
          <Image
            source={require('@/assets/images/bookencyclopedia.png')}
            style={styles.icon}
            resizeMode="contain"
          />
        </TouchableOpacity>

      </View>

    </SafeAreaView>
  );
}


// StyleSheet
// --------------------------------------------------------------------------------------
const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#000000',
  },
  content: {
    flex: 1,
  },
  navigation: {
    height: 65,
    flexDirection: 'row',
    justifyContent: 'space-around',
    alignItems: 'center',
    borderTopWidth: 1,
    borderTopColor: '#fbfbfb',
  },
  navText: {
    color: 'white',
  },
    icon: {
    width: 32,
    height: 32,
    tintColor: "#ffffffc1"
  },
});
